
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")
sys.path.append("../fastai/")


# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-p torch,pandas,numpy -m')


# In[3]:


from pathlib import Path
import itertools
from collections import Counter
from functools import partial, reduce

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from fastai.text import (
    SortishSampler, SortSampler, DataLoader, ModelData, to_gpu
)
from fastai.core import T
from fastai.rnn_reg import EmbeddingDropout
from fastai.text import accuracy
from torch.optim import Adam
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import sentencepiece as spm

from cnlp.fastai_extended import (
    LanguageModelLoader, LanguageModelData, get_transformer_classifier, 
    TransformerTextModel, TextDataset, TransformerLearner, FixedLengthDataLoader
)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


path = Path("../data/cache/lm_unigram_transformer_douban/")
path.mkdir(parents=True, exist_ok=True)


# ## Utility Function(s)

# In[5]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Import And Tokenize Comments and Ratings

# In[6]:


df_ratings = pd.read_csv("../data/ratings_word.csv")
df_ratings.head()


# In[7]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=888)
train_idx, test_idx = next(sss.split(df_ratings, df_ratings.rating))
df_train = df_ratings.iloc[train_idx].copy()
df_test = df_ratings.iloc[test_idx].copy()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=888)
val_idx, test_idx = next(sss.split(df_test, df_test.rating))
df_val = df_test.iloc[val_idx].copy()
df_test = df_test.iloc[test_idx].copy()
del df_ratings


# In[8]:


UNK = 0
BEG = 1
EMB_DIM = 300


# In[37]:


sp = spm.SentencePieceProcessor()
sp.Load("../data/rating_unigram_model.model")


# ### Use the Refitted Vocabulary
# #### Investigate Vocabulary Differences

# In[16]:


itos_orig = []
with open("../data/unigram_model.vocab", mode="r", encoding="utf-8") as f:
    for line in f.readlines():
        itos_orig.append(line.split("\t")[0])
itos = []
with open("../data/rating_unigram_model.vocab", mode="r", encoding="utf-8") as f:
    for line in f.readlines():
        itos.append(line.split("\t")[0])
n_toks = len(itos)
n_toks


# In[10]:


itos[:5]


# In[11]:


mapping = {s: idx for idx, s in enumerate(itos)}
mapping_orig = {s: idx for idx, s in enumerate(itos_orig)}


# In[12]:


voc_diff = set(itos) - set(itos_orig)
print(len(voc_diff), len(itos))
sorted([(x, mapping[x]) for x in list(voc_diff)], key=lambda x: x[1], reverse=True)[:50]


# #### Tokenize

# In[14]:


results = []
tokens_train, tokens_val, tokens_test = [], [], []
for df, tokens in zip((df_train, df_val, df_test), (tokens_train, tokens_val, tokens_test)) :
    for i, row in tqdm_notebook(df.iterrows(), total=df.shape[0]):
        tokens.append(np.array([BEG] + sp.EncodeAsIds(row["comment"])))
assert len(tokens_train) == df_train.shape[0]        


# In[22]:


joblib.dump([n_toks, tokens_train, tokens_val, tokens_test], "../data/cache/rating_unigram_tokens.pkl")


# In[15]:


tokens_val[0]


# In[16]:


df_val.iloc[0]


# #### Prepare the embedding matrix

# In[18]:


MODEL_PATH = "../data/cache/lm_unigram_transformer/models/lm_transformer.h5"
weights = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
assert weights['0.embed.weight'].shape[1] == EMB_DIM
weights['0.embed.weight'].shape


# In[19]:


new_matrix = np.zeros((n_toks + 200, EMB_DIM))
hits = 0
for i, w in enumerate(itos):
    if w in mapping_orig:
        new_matrix[i] = weights['0.embed.weight'][mapping_orig[w]]
        hits += 1
new_matrix[BEG, :] = 0
hits, hits *100 / len(itos[3:])


# In[20]:


new_matrix[n_toks:, :] = weights['0.embed.weight'][-200:, :]


# In[21]:


weights['0.embed.weight'] = T(new_matrix)
weights['1.weight'] = T(np.copy(new_matrix)[:-200, :])
weights['0.embed.weight'].shape


# ## Languange Model

# In[21]:


bs = 32
bptt = 200
target_length = 195
trn_dl = LanguageModelLoader(np.concatenate(tokens_train), bs, bptt, target_length=target_length, batch_first=True)
val_dl = LanguageModelLoader(np.concatenate(tokens_val), bs, bptt, target_length=target_length, batch_first=True)


# In[22]:


model_data = LanguageModelData(
    path, pad_idx=2, n_tok=n_toks, trn_dl=trn_dl, val_dl=val_dl, bs=bs, bptt=bptt
)


# In[23]:


learner = model_data.get_transformer_model(
    partial(Adam, betas=(0.8, 0.999)),
    max_seq_len=trn_dl.max_possible_seq_len,
    emb_sz=EMB_DIM,
    n_head=12,
    n_layer=3,
    embd_pdrop=0.05,
    attn_pdrop=0.1,
    resid_pdrop=0.1
)


# In[24]:


learner.model.load_state_dict(weights)
assert torch.equal(learner.model[0].embed.weight[:-200, :], learner.model[1].weight)


# In[25]:


next(iter(trn_dl))[0].size()


# In[26]:


learner.get_layer_groups()


# In[39]:


from fastai.core import set_trainable
set_trainable(learner.model, False)
set_trainable(learner.get_layer_groups()[0], True)
set_trainable(learner.get_layer_groups()[-1], True)
assert learner.model[0].blocks[0].trainable == False
assert learner.model[0].blocks[0].attn.c_proj.weight.requires_grad == False
assert learner.model[1].weight.requires_grad == True
assert learner.model[1].trainable == True
assert learner.model[0].embed.trainable == True
assert learner.model[0].embed.weight.requires_grad == True


# In[41]:


lr=1e-3
lrs = lr
learner.fit(lrs/2, 1, wds=0, use_clr=(32,2), cycle_len=1)


# In[42]:


learner.save('lm_last_ft')


# In[43]:


learner.unfreeze()
learner.clip = 25
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


# In[44]:


learner.sched.plot()


# In[46]:


get_ipython().run_line_magic('pinfo', 'learner.fit')


# In[47]:


lr = 1e-4
lrs = lr
learner.fit(lrs, n_cycle=1, wds=0, use_clr=(20,4), cycle_len=10)


# In[48]:


learner.save_encoder("lm1_enc")


# In[49]:


learner.save("lm1")


# In[50]:


del learner


# ## 3-class Classifier
# As in https://zhuanlan.zhihu.com/p/27198713

# ### Full Dataset (v1)

# In[10]:


for df in (df_train, df_val, df_test):
    df["label"] = (df["rating"] >= 3) * 1
    df.loc[df.rating == 3, "label"] = 1
    df.loc[df.rating > 3, "label"] = 2


# In[18]:


df_train.label.value_counts()


# In[19]:


bs = 64
trn_ds = TextDataset(tokens_train, df_train.label.values)
val_ds = TextDataset(tokens_val, df_val.label.values)
trn_samp = SortishSampler(tokens_train, key=lambda x: len(tokens_train[x]), bs=bs//2)
val_samp = SortSampler(tokens_val, key=lambda x: len(tokens_val[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=False, num_workers=1, pad_idx=2, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=False, num_workers=1, pad_idx=2, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[20]:


model= get_transformer_classifier(
    n_tok=n_toks, 
    emb_sz=EMB_DIM, 
    n_head=12, 
    n_layer=3, 
    n_ctx=200,
    max_seq_len=100,
    clf_layers=[EMB_DIM, 50, 3],
    pad_token=2,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    clf_pdrop=[0.5, 0.1],
    afn="gelu"
)    


# In[21]:


learn = TransformerLearner(
    model_data, 
    TransformerTextModel(to_gpu(model)), 
    opt_fn=partial(torch.optim.Adam, betas=(0.9, 0.999)))
learn.clip=25
learn.metrics = [accuracy]
learn.load_encoder('lm1_enc')


# In[22]:


lrs = np.array([5e-5, 1e-4, 2e-4, 5e-4, 2e-3])
learn.freeze_to(-1)
learn.lr_find(lrs/1000)
learn.sched.plot()


# In[23]:


lrs = np.array([5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[24]:


learn.save('clas_0')


# In[25]:


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[26]:


learn.save('clas_1')


# In[27]:


learn.unfreeze()
learn.fit(lrs, 1, wds=0, cycle_len=14, use_clr=(32,10))


# In[ ]:


learn.save("clas_full")


# #### Evaluate

# In[38]:


learn.load("clas_full")
learn.model.reset()
_ = learn.model.eval()


# In[40]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(val_dl):
    preds.append(np.argmax(learn.model(x).cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[41]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[42]:


pd.Series(ys).value_counts()


# In[43]:


pd.Series(preds).value_counts()


# In[44]:


np.sum(ys==preds) / ys.shape[0]


# In[45]:


np.where(ys==0)


# In[46]:


tokens_val[176196]


# In[47]:


sp.DecodeIds(tokens_val[176196].tolist()), df_val["comment"].iloc[176196]


# In[48]:


def get_prediction(texts):
    input_tensor = T(np.array([1] + sp.EncodeAsIds(texts))).unsqueeze(1)
    return learn.model(input_tensor)[0].data.cpu().numpy()


# In[49]:


get_prediction("看 了 快 一半 了 才 发现 是 mini 的 广告")


# In[50]:


get_prediction("妈蛋 ， 简直 太 好看 了 。 最后 的 DJ battle 部分 ， 兴奋 的 我 ， 简直 想 从 座位 上 站 起来 一起 扭")


# In[51]:


get_prediction("说 实话 我 没 怎么 认真 看 ， 电影院 里 的 熊 孩子 太 闹腾 了 ， 前面 的 小奶娃 还 时不时 站 "
               "在 老爸 腿上 蹦迪 ， 观影 体验 极差 ， 不过 小朋友 应该 挺 喜欢 的")


# In[52]:


get_prediction("这 电影 太 好笑 了 ， 说好 的 高科技 人才 研制 的 产品 永远 在 关键 时候 失灵 "
               "； 特地 飞 到 泰国 请来 救援 人才 ， 大家 研究 出 的 方法 每次 都是 先 给 鲨鱼 "
               "当 诱饵 … … 显然 这样 的 对战 坚持不了 多久 ， 只能 赶紧 让 鲨鱼 输 了 。")


# In[53]:


get_prediction("太 接地气 了 ， 在 三亚 煮饺子 式 的 景区 海域 ， 冒出来 一条 大 鲨鱼 "
               "… … 爽点 也 很 密集 ， 郭达森 与 李冰冰 的 CP 感 不错 ， 编剧 果然 是 "
               "老外 ， 中文 台词 有点 尬 。")


# In[54]:


get_prediction("李冰冰 的 脸 真的 很 紧绷 ， 比 鲨鱼 的 脸 还 绷 。")


# In[55]:


get_prediction("太 难 了 。 。 。")


# In[56]:


get_prediction("把 我 基神 写成 智障 ， 辣鸡 mcu")


# In[57]:


get_prediction("鲨鱼 部分 还是 不错 的 ， 尤其 是 中段 第一次 出海 捕鲨 非常 刺激 ， 其后 急速 下滑 ， "
               "三亚 那 部分 拍得 是什么 鬼 。 。 。 爆米花 片 可以 适度 的 蠢 ， 但 人类 反派 炸鲨 "
               "和 直升机 相撞 部分 简直 蠢得 太过份 了 吧 ？ 另外 充满 硬 加戏 视感 的 尴尬 感情戏 "
               "把 节奏 也 拖垮 了 ， 明明 可以 更 出色 ， 却 很遗憾 地 止步 在 马马虎虎 的 水平 。 6 / 10")


# In[58]:


get_prediction("老冰冰 真的 很努力 ！ 为 老冰冰 实现 了 她 的 好莱坞 女主梦 鼓掌 . . .")


# In[59]:


get_prediction("结局 简直 丧 出 天际 ！ 灭霸 竟然 有 内心戏 ！ 全程 下来 美队 "
               "和 钢铁侠 也 没 见上 一面 ， 我 还 以为 在 世界 末日 前 必然 "
               "要 重修 旧好 了 ！ ")


# In[60]:


get_prediction("太 烂 了 ， 难看 至极 。")


# In[61]:


get_prediction("看完 之后 很 生气 ！ 剧情 太差 了")


# In[62]:


get_prediction("关键点 都 好傻 ， 我 知道 你 要拍 续集 ， "
               "我 知道 未来 可以 被 重写 ， 但 那 一拳 真的 有点 傻 。")


# In[63]:


get_prediction("好了 可以 了 。 再也 不看 Marvel 了 。 我 努力 过 了 。 实在 是 。 。 啥呀 这是 。 🙄️")


# In[64]:


get_prediction("还 我 电影票 14 元")


# In[65]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[66]:


plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[67]:


precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# In[68]:


test_ds = TextDataset(tokens_test, df_test.label.values)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=False, num_workers=1, pad_idx=0, sampler=test_samp)


# In[71]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(test_dl):
    preds.append(np.argmax(learn.model(x).cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[72]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[73]:


np.sum(ys==preds) / ys.shape[0]


# In[74]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[75]:


plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[77]:


from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# ### Smaller Dataset 

# In[9]:


n_toks, tokens_train, tokens_val, tokens_test = joblib.load("../data/cache/rating_unigram_tokens.pkl")


# In[10]:


for df in (df_train, df_val, df_test):
    df["label"] = (df["rating"] >= 3) * 1
    df.loc[df.rating == 3, "label"] = 1
    df.loc[df.rating > 3, "label"] = 2


# In[11]:


df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)


# In[12]:


df_train_small = pd.concat([
    df_train[df_train.label==0].sample(15000),
    df_train[df_train.label==1].sample(15000),
    df_train[df_train.label==2].sample(15000)
], axis=0)
df_val_small = pd.concat([
    df_val[df_val.label==0].sample(5000),
    df_val[df_val.label==1].sample(5000),
    df_val[df_val.label==2].sample(5000)
], axis=0)


# In[13]:


np.array(df_train_small.index)


# In[19]:


bs = 64
tokens_train_small = np.array(tokens_train)[np.array(df_train_small.index)]
tokens_val_small = np.array(tokens_val)[np.array(df_val_small.index)]
trn_ds = TextDataset(tokens_train_small, df_train_small.label.values, max_seq_len=200)
val_ds = TextDataset(tokens_val_small, df_val_small.label.values, max_seq_len=200)
trn_samp = SortishSampler(tokens_train_small, key=lambda x: len(tokens_train_small[x]), bs=bs//2)
val_samp = SortSampler(tokens_val_small, key=lambda x: len(tokens_val_small[x]))
trn_dl = FixedLengthDataLoader(trn_ds, batch_size=bs//2, seq_length=200, transpose=False, num_workers=1, pad_idx=2, sampler=trn_samp)
val_dl = FixedLengthDataLoader(val_ds, batch_size=bs, seq_length=200, transpose=False, num_workers=1, pad_idx=2, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[20]:


model= get_transformer_classifier(
    n_tok=n_toks, 
    emb_sz=EMB_DIM, 
    n_head=12, 
    n_layer=3, 
    n_ctx=200,
    clf_layers=[EMB_DIM, 50, 3],
    pad_token=2,
    embd_pdrop=0.05,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    clf_pdrop=[0.5, 0.2],
    afn="gelu"
)


# In[21]:


learn = TransformerLearner(
    model_data, 
    TransformerTextModel(to_gpu(model)), 
    opt_fn=partial(torch.optim.Adam, betas=(0.9, 0.999)))
learn.clip=25
learn.metrics = [accuracy]
learn.load_encoder('lm1_enc')


# In[50]:


lrs = np.array([1e-5, 1e-4, 1e-4, 1e-4, 2e-4])
learn.freeze_to(-1)
learn.lr_find(lrs/100)
learn.sched.plot()


# In[22]:


learn.freeze_to(-1)
lrs = np.array([1e-5, 1e-4, 2e-4, 5e-4, 1e-3])
learn.fit(lrs, 1, wds=1e-6, cycle_len=1, use_clr=(8,3), use_wd_sched=True)


# In[18]:


# Debug Purpose Only
learn.model.eval()
preds, ys, logloss, batch_cnts = [], [], [], []
with torch.set_grad_enabled(False):
    for x, y in tqdm_notebook(model_data.val_dl):
        tmp = learn.model(x).cpu()
        batch_cnts.append(x.shape[0])
        logloss.append(F.cross_entropy(tmp, y.cpu()).data.numpy())
        preds.append(np.argmax(tmp.cpu().data.numpy(), axis=1))
        ys.append(y.cpu().numpy())
logloss = np.array(logloss)
preds = np.concatenate(preds)
ys = np.concatenate(ys)
batch_cnts = np.array(batch_cnts)
print(preds.shape, ys.shape)
 np.average(logloss, 0, weights=batch_cnts), np.mean(preds==ys)


# In[23]:


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=1e-6, cycle_len=1, use_clr=(8,3), use_wd_sched=True)


# In[24]:


learn.unfreeze()
learn.fit(lrs, 1, wds=1e-6, cycle_len=10, use_clr=(32,10), use_wd_sched=True)


# In[25]:


learn.save("clas_small_full")


# In[29]:


learn.model.eval()
preds, ys = [], []
for x, y in val_dl:
    preds.append(np.argmax(learn.model(x).cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[30]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[31]:


np.sum(preds==ys) / preds.shape[0]


# In[32]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[33]:


test_ds = TextDataset(tokens_test, df_test.label.values, max_seq_len=200)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=False, num_workers=1, pad_idx=2, sampler=test_samp)


# In[34]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(test_dl):
    preds.append(np.argmax(learn.model(x).cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[35]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
print(preds.shape, ys.shape)
np.sum(preds==ys) / preds.shape[0]


# In[36]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[37]:


precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# ## Regressor

# In[9]:


n_toks, tokens_train, tokens_val, tokens_test = joblib.load("../data/cache/rating_unigram_tokens.pkl")


# In[10]:


bs = 128
trn_ds = TextDataset(tokens_train, df_train.rating.values.astype("float32"), max_seq_len=200)
val_ds = TextDataset(tokens_val, df_val.rating.values.astype("float32"), max_seq_len=200)
trn_samp = SortishSampler(tokens_train, key=lambda x: len(tokens_train[x]), bs=bs//2)
val_samp = SortSampler(tokens_val, key=lambda x: len(tokens_val[x]))
trn_dl = FixedLengthDataLoader(trn_ds, seq_length=200, batch_size=bs//2, transpose=False, num_workers=1, pad_idx=2, sampler=trn_samp)
val_dl = FixedLengthDataLoader(val_ds, seq_length=200, batch_size=bs, transpose=False, num_workers=1, pad_idx=2, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[11]:


tmp = next(iter(trn_dl))
tmp[0].size()


# In[12]:


for x, _ in trn_dl:
    if x.size(1) != 200:
        print(x.size())
    assert x.size(1) == 200


# In[13]:


tmp[0][2, :]


# In[14]:


model= get_transformer_classifier(
    n_tok=n_toks, 
    emb_sz=EMB_DIM, 
    n_head=12, 
    n_layer=3, 
    n_ctx=200,
    clf_layers=[EMB_DIM, 100, 1],
    pad_token=2,
    embd_pdrop=0.05,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    clf_pdrop=[0.5, 0.2],
    afn="gelu"
)    


# In[15]:


class TruncatedTransformerRegLearner(TransformerLearner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return lambda x, y: F.mse_loss(x[:, 0], y)


# In[16]:


learn = TruncatedTransformerRegLearner(
    model_data, 
    TransformerTextModel(to_gpu(model)), 
    opt_fn=partial(torch.optim.Adam, betas=(0.9, 0.999)))
learn.clip=25
learn.load_encoder('lm1_enc')


# In[33]:


learn.model[1]


# In[75]:


lrs = np.array([5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
learn.freeze_to(-1)
learn.lr_find(lrs/1000)
learn.sched.plot()


# In[34]:


lrs = np.array([5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
learn.freeze_to(-1)
get_ipython().run_line_magic('time', 'learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))')
learn.save('reg_0')


# In[35]:


learn.freeze_to(-3)
get_ipython().run_line_magic('time', 'learn.fit(lrs, 1, wds=1e-6, cycle_len=1, use_clr=(8,3), use_wd_sched=True)')


# In[36]:


learn.unfreeze()
get_ipython().run_line_magic('time', 'learn.fit(lrs, 1, wds=1e-6, cycle_len=12, use_clr=(32,5), use_wd_sched=True)')
learn.save('reg_full')


# In[21]:


# Export Model
torch.save(learn.model, path / "sentiment_model.pth")


# In[17]:


learn.load('reg_full')


# ### Evaluation

# In[18]:


def get_preds(data_loader):
    learn.model.eval()       
    preds, ys = [], []
    for x, y in tqdm_notebook(data_loader):   
        with torch.set_grad_enabled(False):
            preds.append(learn.model(x).cpu().data.numpy()[:, 0])
            ys.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return ys, preds


# In[19]:


ys, preds = get_preds(val_dl)
print("(Validation set):", preds.shape, ys.shape)
np.sum(np.square(preds - ys)) / preds.shape[0]


# In[20]:


pd.Series(preds).describe()


# In[21]:


test_ds = TextDataset(tokens_test, df_test.rating.values, max_seq_len=200)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=False, num_workers=1, pad_idx=2, sampler=test_samp)


# In[22]:


ys, preds = get_preds(test_dl)
print("(Test set):", preds.shape, ys.shape)
np.sum(np.square(preds - ys)) / preds.shape[0]


# In[23]:


pd.Series(ys).describe()


# In[24]:


np.sum(np.square(preds - ys)) / preds.shape[0]


# In[25]:


preds = np.clip(preds, 1, 5)
np.sum(np.square(preds - ys)) / preds.shape[0]


# In[28]:


# Save predictions
df_test.loc[df_test.iloc[list(iter(test_samp))].index, "preds"] = preds
# df_test.to_csv(path / "df_test.csv.gz", index=False, compression="gzip")
df_test.head()


# In[29]:


df_test.sample(20)


# In[30]:


np.sum(np.square(df_test.rating.values - df_test.preds.values)) / preds.shape[0]


# In[31]:


preds_class = np.round(preds)


# In[32]:


cnf_matrix = confusion_matrix(ys, preds_class)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[33]:


from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(ys, preds_class)
for i in range(5):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# In[34]:


def get_prediction(texts):
    input_tensor = T(np.array([1] + sp.EncodeAsIds(texts))).unsqueeze(1)
    return learn.model(input_tensor)[0].data.cpu().numpy()[0, 0]


# In[38]:


get_prediction("看 了 快 一半 了 才 发现 是 mini 的 广告")


# In[ ]:


get_prediction("妈蛋 ， 简直 太 好看 了 。 最后 的 DJ battle 部分 ， 兴奋 的 我 ， 简直 想 从 座位 上 站 起来 一起 扭")


# In[ ]:


get_prediction("关键点 都 好傻 ， 我 知道 你 要拍 续集 ， "
               "我 知道 未来 可以 被 重写 ， 但 那 一拳 真的 有点 傻 。")


# In[ ]:


get_prediction("李冰冰 的 脸 真的 很 紧绷 ， 比 鲨鱼 的 脸 还 绷 。")


# In[ ]:


get_prediction("太 烂 了 ， 难看 至极 。")


# In[ ]:


get_prediction("看完 之后 很 生气 ！ 剧情 太差 了")


# In[ ]:


get_prediction("好了 可以 了 。 再也 不看 Marvel 了 。 我 努力 过 了 。 实在 是 。 。 啥呀 这是 。 🙄️")


# In[ ]:


get_prediction("还 我 电影票 14 元")

