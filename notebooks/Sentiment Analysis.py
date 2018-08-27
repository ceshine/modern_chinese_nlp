
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")


# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-ptorch,pandas,numpy -m')


# In[3]:


from pathlib import Path
import itertools
from collections import Counter
from functools import partial, reduce

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from fastai.text import (
    TextDataset, SortishSampler, SortSampler, DataLoader, ModelData, get_rnn_classifier, seq2seq_reg, 
    RNN_Learner, TextModel, to_gpu, LanguageModelLoader, LanguageModelData
)
from fastai.core import T
from fastai.text import accuracy
from fastai.rnn_reg import EmbeddingDropout
from torch.optim import Adam
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


path = Path("../data/cache/lm_douban/")
path.mkdir(parents=True, exist_ok=True)


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


UNK = 2
BEG = 1
MIN_FREQ = 200
VOC_SIZE = 10000
EMB_DIM = 300


# In[7]:


mapping_orig = joblib.load("../data/mapping.pkl")
df_ratings = pd.read_csv("../data/ratings.csv")
df_ratings.head()


# ### Refit the Vocabulary

# In[8]:


cnt = Counter(itertools.chain.from_iterable(df_ratings["comment"]))
cnt.most_common(10)


# In[9]:


cnt.most_common(VOC_SIZE)[-10:]


# In[135]:


mapping = {
    word: token + 3 for token, (word, freq) in enumerate(
        cnt.most_common(VOC_SIZE))
    if freq > MIN_FREQ
}
joblib.dump(mapping, "data/mapping_sent.pkl")
n_toks = len(mapping)
itos = ["pad", "BEG", "UNK"] + [0] *  n_toks
for k, v in mapping.items():
    itos[v] = k
n_toks = len(itos) + 1
len(itos)


# In[11]:


itos[-10:]


# In[12]:


voc_diff = set(mapping.keys()) - set(mapping_orig.keys())
sorted([(x, mapping[x]) for x in list(voc_diff)], key=lambda x: x[1], reverse=True)[:20]


# In[14]:


len(voc_diff)


# ### Tokenize

# In[15]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=888)
train_idx, test_idx = next(sss.split(df_ratings, df_ratings.rating))
df_train = df_ratings.iloc[train_idx].copy()
df_test = df_ratings.iloc[test_idx].copy()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=888)
val_idx, test_idx = next(sss.split(df_test, df_test.rating))
df_val = df_test.iloc[val_idx].copy()
df_test = df_test.iloc[test_idx].copy()
del df_ratings


# In[16]:


df_test.iloc[0]["comment"],[mapping.get(x, 1) for x in df_test.iloc[0]["comment"]]


# In[17]:


results = []
tokens_train, tokens_val, tokens_test = [], [], []
for df, tokens in zip((df_train, df_val, df_test), (tokens_train, tokens_val, tokens_test)) :
    for i, row in tqdm_notebook(df.iterrows(), total=df.shape[0]):
        tokens.append(np.array([BEG] + [mapping.get(x, UNK) for x in row["comment"]]))


# In[18]:


assert len(tokens_train) == df_train.shape[0]


# ### Prepare the embedding matrix

# In[19]:


MODEL_PATH = "../data/cache/lm/models/lm_lstm.h5"
weights = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
assert weights['0.encoder.weight'].shape[1] == EMB_DIM
weights['0.encoder.weight'].shape


# In[20]:


new_matrix = np.zeros((n_toks, EMB_DIM))
hits = 0
for i, w in enumerate(itos):
    if w in mapping_orig:
        new_matrix[i] = weights['0.encoder.weight'][mapping_orig[w]]
        hits += 1
hits, hits *100 / len(itos[3:])


# In[21]:


weights['0.encoder.weight'] = T(new_matrix)
weights['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_matrix))
weights['1.decoder.weight'] = T(np.copy(new_matrix))


# ## Languange Model

# In[22]:


bs = 64
bptt = 50
trn_dl = LanguageModelLoader(np.concatenate(tokens_train), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(tokens_val), bs, bptt)


# In[23]:


model_data = LanguageModelData(path, 1, n_toks, trn_dl, val_dl, bs=bs, bptt=bptt)


# In[24]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7
opt_fn = partial(torch.optim.Adam, betas=(0.8, 0.99))


# In[25]:


learner = model_data.get_model(opt_fn, EMB_DIM, 500, 3, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[26]:


learner.model.load_state_dict(weights)


# In[27]:


lr=1e-3
lrs = lr
learner.fit(lrs/2, 1, wds=1e-7, use_clr=(32,2), cycle_len=1)


# In[28]:


learner.save('lm_last_ft')


# In[29]:


learner.unfreeze()
learner.clip = 25
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


# In[30]:


learner.sched.plot()


# In[31]:


lr = 3e-3
lrs = lr
learner.fit(lrs, 1, wds=1e-7, use_clr=(20,5), cycle_len=10)


# In[32]:


learner.save_encoder("lm1_enc")


# In[33]:


learner.save("lm1")


# In[34]:


del learner


# ## 3-class Classifier
# As in https://zhuanlan.zhihu.com/p/27198713

# In[35]:


for df in (df_train, df_val, df_test):
    df["label"] = (df["rating"] >= 3) * 1
    df.loc[df.rating == 3, "label"] = 1
    df.loc[df.rating > 3, "label"] = 2


# In[36]:


df_train.label.value_counts()


# In[37]:


bs = 64
trn_ds = TextDataset(tokens_train, df_train.label.values)
val_ds = TextDataset(tokens_val, df_val.label.values)
trn_samp = SortishSampler(tokens_train, key=lambda x: len(tokens_train[x]), bs=bs//2)
val_samp = SortSampler(tokens_val, key=lambda x: len(tokens_val[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=0, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[38]:


dps = np.array([0.4,0.5,0.05,0.3,0.4]) * 0.5
opt_fn = partial(torch.optim.Adam, betas=(0.7, 0.99))
bptt = 50


# In[39]:


model = get_rnn_classifier(bptt, bptt*2, 3, n_toks, emb_sz=EMB_DIM, n_hid=500, n_layers=3, pad_token=0,
          layers=[EMB_DIM*3, 50, 3], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[40]:


learn = RNN_Learner(model_data, TextModel(to_gpu(model)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = [accuracy]
learn.load_encoder('lm1_enc')


# In[41]:


learn.freeze_to(-1)
learn.lr_find(lrs/1000)
learn.sched.plot()


# In[42]:


lr=2e-4
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[43]:


learn.save('clas_0')


# In[44]:


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[45]:


learn.save('clas_1')


# In[46]:


learn.unfreeze()
learn.fit(lrs, 1, wds=0, cycle_len=14, use_clr=(32,10))


# In[47]:


learn.save("clas_full")


# ### Evaluate

# In[50]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(val_dl):
    preds.append(np.argmax(learn.model(x)[0].cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[51]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[52]:


pd.Series(ys).value_counts()


# In[53]:


pd.Series(preds).value_counts()


# In[54]:


np.sum(ys==preds) / ys.shape[0]


# In[55]:


itos = ["pad", "BEG", "UNK"] + [0] *  n_toks
for k, v in mapping.items():
    itos[v+1] = k


# In[56]:


np.where(ys==0)


# In[57]:


"".join([itos[x] for x in tokens_val[176204]])


# In[74]:


def get_prediction(texts):
    input_tensor = T(np.array([1] + [mapping.get(x, UNK) for x in texts])).unsqueeze(1)
    return learn.model(input_tensor)[0].data.cpu().numpy()


# In[75]:


get_prediction("看了快一半了才发现是mini的广告")


# In[76]:


get_prediction("妈蛋，简直太好看了。最后的DJ battle部分，兴奋的我，简直想从座位上站起来一起扭")


# In[77]:


get_prediction("说实话我没怎么认真看，电影院里的熊孩子太闹腾了，前面的小奶娃还时不时站在老爸腿上蹦迪，观影体验极差，不过小朋友应该挺喜欢的")


# In[78]:


get_prediction("这电影太好笑了，说好的高科技人才研制的产品永远在关键时候失灵；特地飞到泰国请来救援人才，大家研究出的方法每次都是先给鲨鱼当诱饵……显然这样的对战坚持不了多久，只能赶紧让鲨鱼输了。")


# In[79]:


get_prediction("太接地气了，在三亚煮饺子式的景区海域，冒出来一条大鲨鱼……爽点也很密集，郭达森与李冰冰的CP感不错，编剧果然是老外，中文台词有点尬。")


# In[80]:


get_prediction("李冰冰的脸真的很紧绷，比鲨鱼的脸还绷。")


# In[81]:


get_prediction("太难了。。。")


# In[82]:


get_prediction("把我基神写成智障，辣鸡mcu")


# In[83]:


get_prediction("鲨鱼部分还是不错的，尤其是中段第一次出海捕鲨非常刺激，其后急速下滑，三亚那部分拍得是什么鬼。。。爆米花片可以适度的蠢，但人类反派炸鲨和直升机相撞部分简直蠢得太过份了吧？另外充满硬加戏视感的尴尬感情戏把节奏也拖垮了，明明可以更出色，却很遗憾地止步在马马虎虎的水平。6/10")


# In[84]:


get_prediction("老冰冰真的很努力！为老冰冰实现了她的好莱坞女主梦鼓掌...")


# In[85]:


get_prediction("结局简直丧出天际！灭霸竟然有内心戏！全程下来美队和钢铁侠也没见上一面，我还以为在世界末日前必然要重修旧好了！")


# In[86]:


get_prediction("太烂了，难看至极。")


# In[87]:


get_prediction("看完之后很生气！剧情太差了")


# In[88]:


get_prediction("关键点都好傻，我知道你要拍续集，我知道未来可以被重写， 但那一拳真的有点傻。")


# In[89]:


get_prediction("好了可以了。再也不看Marvel了。我努力过了。实在是。。啥呀这是。🙄️")


# In[90]:


get_prediction("还我电影票14元")


# In[91]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[92]:


plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[93]:


from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# In[94]:


test_ds = TextDataset(tokens_test, df_test.label.values)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=test_samp)


# In[95]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(test_dl):
    preds.append(np.argmax(learn.model(x)[0].cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[96]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[97]:


np.sum(ys==preds) / ys.shape[0]


# In[98]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[99]:


plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[100]:


from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# ### Smaller Dataset 

# In[101]:


df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)


# In[102]:


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


# In[103]:


np.array(df_train_small.index)


# In[104]:


bs = 64
tokens_train_small = np.array(tokens_train)[np.array(df_train_small.index)]
tokens_val_small = np.array(tokens_val)[np.array(df_val_small.index)]
trn_ds = TextDataset(tokens_train_small, df_train_small.label.values)
val_ds = TextDataset(tokens_val_small, df_val_small.label.values)
trn_samp = SortishSampler(tokens_train_small, key=lambda x: len(tokens_train_small[x]), bs=bs//2)
val_samp = SortSampler(tokens_val_small, key=lambda x: len(tokens_val_small[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=0, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[105]:


dps = np.array([0.4,0.5,0.05,0.3,0.4])
opt_fn = partial(torch.optim.Adam, betas=(0.7, 0.99))
bptt = 50


# In[107]:


model = get_rnn_classifier(bptt, bptt*2, 3, n_toks, emb_sz=EMB_DIM, n_hid=500, n_layers=3, pad_token=0,
          layers=[EMB_DIM*3, 50, 3], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[108]:


learn = RNN_Learner(model_data, TextModel(to_gpu(model)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = [accuracy]
learn.load_encoder('lm1_enc')


# In[109]:


learn.freeze_to(-1)
learn.lr_find(lrs/100)
learn.sched.plot()


# In[110]:


lr=2e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[111]:


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))


# In[112]:


learn.unfreeze()
learn.fit(lrs, 1, wds=0, cycle_len=14, use_clr=(32,10))


# In[113]:


learn.save("clas_small_full")


# In[114]:


learn.model.eval()
preds, ys = [], []
for x, y in val_dl:
    preds.append(np.argmax(learn.model(x)[0].cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[115]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[116]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[117]:


test_ds = TextDataset(tokens_test, df_test.label.values)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=test_samp)


# In[118]:


learn.model.eval()
preds, ys = [], []
for x, y in tqdm_notebook(test_dl):
    preds.append(np.argmax(learn.model(x)[0].cpu().data.numpy(), axis=1))
    ys.append(y.cpu().numpy())


# In[119]:


preds = np.concatenate(preds)
ys = np.concatenate(ys)
preds.shape, ys.shape


# In[120]:


np.sum(preds==ys) / preds.shape[0]


# In[121]:


cnf_matrix = confusion_matrix(ys, preds)
np.set_printoptions(precision=2)
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2], normalize=True,
    title='Confusion matrix, without normalization')


# In[122]:


precision, recall, fscore, support = precision_recall_fscore_support(ys, preds)
for i in range(3):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# ## Regressor

# In[123]:


bs = 64
trn_ds = TextDataset(tokens_train, df_train.rating.values.astype("float32"))
val_ds = TextDataset(tokens_val, df_val.rating.values.astype("float32"))
trn_samp = SortishSampler(tokens_train, key=lambda x: len(tokens_train[x]), bs=bs//2)
val_samp = SortSampler(tokens_val, key=lambda x: len(tokens_val[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=0, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=val_samp)
model_data = ModelData(path, trn_dl, val_dl)


# In[124]:


dps = np.array([0.4,0.5,0.05,0.3,0.4]) * 0.5
opt_fn = partial(torch.optim.Adam, betas=(0.7, 0.99))
bptt = 50


# In[125]:


model = get_rnn_classifier(bptt, bptt*2, 3, n_toks, emb_sz=EMB_DIM, n_hid=500, n_layers=3, pad_token=0,
          layers=[EMB_DIM*3, 50, 1], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[126]:


class RNN_RegLearner(RNN_Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return lambda x, y: F.mse_loss(x[:, 0], y)


# In[127]:


learn = RNN_RegLearner(model_data, TextModel(to_gpu(model)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = []
learn.load_encoder('lm1_enc')


# In[128]:


lr=2e-4
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])


# In[129]:


learn.freeze_to(-1)
learn.lr_find(lrs/1000)
learn.sched.plot()


# In[130]:


learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))
learn.save('reg_0')


# In[131]:


learn.freeze_to(-2)
learn.fit(lrs, 1, wds=0, cycle_len=1, use_clr=(8,3))
learn.save('reg_1')


# In[132]:


learn.unfreeze()
learn.fit(lrs, 1, wds=0, cycle_len=14, use_clr=(32,10))
learn.save('reg_full')


# In[133]:


# Export Model
torch.save(learn.model, path / "sentiment_model.pth")


# In[134]:


learn.load('reg_full')


# ### Evaluation

# In[138]:


test_ds = TextDataset(tokens_test, df_test.rating.values)
test_samp = SortSampler(tokens_test, key=lambda x: len(tokens_test[x]))
test_dl = DataLoader(test_ds, bs, transpose=True, num_workers=1, pad_idx=0, sampler=test_samp)


# In[139]:


def get_preds(data_loader):
    learn.model.eval()
    learn.model.reset()         
    preds, ys = [], []
    for x, y in tqdm_notebook(data_loader):   
        preds.append(learn.model(x)[0].cpu().data.numpy()[:, 0])
        ys.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return ys, preds
ys, preds = get_preds(val_dl)
preds.shape, ys.shape


# In[140]:


pd.Series(ys).describe()


# In[141]:


pd.Series(ys).describe()


# In[142]:


np.sum(np.square(preds - ys)) / preds.shape[0]


# In[143]:


preds = np.clip(preds, 1, 5)
np.sum(np.square(preds - ys)) / preds.shape[0]


# In[144]:


# Save predictions
df_val.loc[df_val.iloc[list(iter(val_samp))].index, "preds"] = preds
df_val.to_csv(path / "df_val.csv.gz", index=False, compression="gzip")
df_val.head()


# In[145]:


np.sum(np.square(df_val.rating.values - df_val.preds.values)) / preds.shape[0]


# In[146]:


ys, preds = get_preds(test_dl)
preds.shape, ys.shape


# In[147]:


preds = np.clip(preds, 1, 5)
np.sum(np.square(preds - ys)) / preds.shape[0]


# In[148]:


# Save predictions
df_test.loc[df_test.iloc[list(iter(test_samp))].index, "preds"] = preds
df_test.to_csv(path / "df_test.csv.gz", index=False, compression="gzip")
df_test.head()


# In[149]:


df_test.sample(20)


# In[150]:


np.sum(np.square(df_test.rating.values - df_test.preds.values)) / preds.shape[0]


# In[151]:


preds_class = np.round(preds)


# In[152]:


cnf_matrix = confusion_matrix(ys, preds_class)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=[0, 1, 2],
    title='Confusion matrix, without normalization')


# In[153]:


from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(ys, preds_class)
for i in range(5):
    print(f"Class {i}: P {precision[i]*100:.0f}%, R {recall[i]*100:.0f}%, FS {fscore[i]:.2f}, Support: {support[i]}")


# In[163]:


def get_prediction(texts):
    input_tensor = T(np.array([1] + [mapping.get(x, UNK) for x in texts])).unsqueeze(1)
    return learn.model(input_tensor)[0].data.cpu().numpy()[0, 0]


# In[164]:


get_prediction("看了快一半了才发现是mini的广告")


# In[165]:


get_prediction("妈蛋，简直太好看了。最后的DJ battle部分，兴奋的我，简直想从座位上站起来一起扭")


# In[166]:


get_prediction("说实话我没怎么认真看，电影院里的熊孩子太闹腾了，前面的小奶娃还时不时站在老爸腿上蹦迪，观影体验极差，不过小朋友应该挺喜欢的")


# In[167]:


get_prediction("李冰冰的脸真的很紧绷，比鲨鱼的脸还绷。")


# In[168]:


get_prediction("太烂了，难看至极。")


# In[169]:


get_prediction("还我电影票14元")


# In[170]:


get_prediction("好了可以了。再也不看Marvel了。我努力过了。实在是。。啥呀这是。🙄️")


# In[171]:


get_prediction("把我基神写成智障，辣鸡mcu")

