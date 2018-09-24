
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")


# In[2]:


from pathlib import Path
from functools import partial

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from fastai.text import LanguageModelLoader, LanguageModelData
from fastai.core import T
from fastai.rnn_reg import EmbeddingDropout
from torch.optim import Adam
import torch.nn as nn
import torch
import torch.nn.functional as F

import sentencepiece as spm


# In[3]:


tokens = joblib.load("../data/tokens_unigram.pkl")


# In[4]:


# Filter out empty texts
tokens = [x for x in tokens if x.shape[0] > 0]


# In[5]:


# Set shuffle = False to keep sentences from the same paragraph together
trn_tokens, val_tokens = train_test_split(tokens, test_size=0.2, shuffle=False)
val_tokens, tst_tokens = train_test_split(val_tokens, test_size=0.5, shuffle=False)


# In[6]:


def get_voc_stats(tokens):
    total_tokens = np.sum([x.shape[0] for x in tokens])
    unks = np.sum([np.sum(x == 0) for x in tokens])
    print("Total tokens: %d\nUnknown Percentage: %.2f %%" % (total_tokens, unks * 100 / total_tokens))
get_voc_stats(tokens)


# In[7]:


bptt = 75
batch_size = 64
n_tok = int(np.max([np.max(x) for x in tokens]) + 1)
trn_loader = LanguageModelLoader(
    np.concatenate(trn_tokens), batch_size, bptt)
val_loader = LanguageModelLoader(
    np.concatenate(val_tokens), batch_size, bptt)
tst_loader = LanguageModelLoader(
    np.concatenate(tst_tokens), batch_size, bptt)


# In[8]:


sp = spm.SentencePieceProcessor()
sp.Load("../data/unigram_model.model")


# In[9]:


sp.EncodeAsIds(", 的*")


# In[10]:


np.sum([np.sum(x == 1) for x in tokens]) # <s>


# In[11]:


np.sum([np.sum(x == 2) for x in tokens]) # </s>


# In[12]:


sp.DecodeIds(trn_tokens[0].tolist())


# In[13]:


sp.DecodeIds(trn_tokens[1].tolist())


# In[14]:


from collections import Counter
tmp = []
for i in range(10000):
    for j in range(1, trn_tokens[i].shape[0]):
        if trn_tokens[i][j] == 0:
            tmp.append(trn_tokens[i][j-1])
Counter(tmp).most_common(10)


# In[15]:


from collections import Counter
tmp = []
for i in range(10000):
    for j in range(1, trn_tokens[i].shape[0]-1):
        if trn_tokens[i][j] == 4:
            tmp.append(trn_tokens[i][j+1])
Counter(tmp).most_common(10)


# In[16]:


sp.DecodeIds([4569])


# In[17]:


path = Path("../data/cache/lm_unigram/")
path.mkdir(parents=True, exist_ok=True)
model_data = LanguageModelData(
    path, pad_idx=2, n_tok=n_tok, trn_dl=trn_loader, val_dl=val_loader, test_dl=tst_loader
)


# In[18]:


n_tok


# ### QRNN Model

# In[21]:


drops = np.array([0.05, 0.1, 0.05, 0, 0.1])
learner = model_data.get_model(
    partial(Adam, betas=(0.8, 0.999)),
    emb_sz=300, n_hid=500, n_layers=4,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
    dropoute=drops[3], dropouth=drops[4], qrnn=True
)


# In[22]:


learner.clip = 25.
learner.lr_find(start_lr=1e-5, end_lr=1, linear=False)
learner.sched.plot()


# In[22]:


lrs = 2e-3
learner.fit(lrs, 1, wds=1e-7, use_clr=(50, 3), cycle_len=10, use_wd_sched=True)


# In[23]:


learner.sched.plot_lr()


# In[43]:


lrs = 5e-4
learner.fit(lrs, 1, wds=1e-7, use_clr=(50, 3), cycle_len=10, use_wd_sched=True)


# In[14]:


learner.sched.plot_loss()


# In[44]:


learner.save("lm_qrnn")
learner.save_encoder("lm_qrnn_enc")


# In[ ]:


learner.load("lm_qrnn")


# ### LSTM

# In[19]:


drops = np.array([0.1, 0.1, 0.05, 0, 0.1])
learner = model_data.get_model(
    partial(Adam, betas=(0.8, 0.999)),
    emb_sz=300, n_hid=500, n_layers=3,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
    dropoute=drops[3], dropouth=drops[4], qrnn=False
)


# In[20]:


learner.clip = 25.
learner.lr_find(start_lr=1e-5, end_lr=1, linear=False)
learner.sched.plot()


# In[21]:


lrs = 2e-3
learner.clip = 10.
learner.fit(lrs, 1, wds=1e-7, use_clr=(50, 5), cycle_len=20, use_wd_sched=True)


# In[22]:


learner.sched.plot_lr()


# In[23]:


learner.save("lm_lstm")
learner.save_encoder("lm_lstm_enc")


# In[24]:


tmp_iter = iter(trn_loader)


# In[25]:


next(tmp_iter)[0].shape


# In[26]:


learner.load("lm_lstm")


# ## Test the model

# In[27]:


learner.model.eval()


# ### Next Character Inference

# In[28]:


tokens = sp.EncodeAsIds("德国 是 世界 大国 之 一 ， 其 国内 生产 总 值 以 国际 汇率 计")
tokens


# In[29]:


logits, _, _ = learner.model(T(tokens).unsqueeze(1))
logits.shape


# In[30]:


sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
preds = []
for i in range(1, 4):
      preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
# preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
pd.DataFrame({"orig": sp.EncodeAsPieces("德国 是 世界 大国 之 一 ， 其 国内 生产 总 值 以 国际 汇率 计") + [""], 
              "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2]})


# In[31]:


def eval(texts):
    learner.model[0].reset()
    tokens =sp.EncodeAsIds(texts)
    logits, _, _ = learner.model(T(tokens).unsqueeze(1))
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
    # preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
    return pd.DataFrame({"orig": sp.EncodeAsPieces(texts) + [""], 
                  "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2]})


# In[32]:


eval("在 现代 印刷 媒体 ， 卡通 是 一 种 通常 有 幽默 色")


# In[33]:


eval("对 中国 与 南洋 发动 全面 的 战争 。 1990 年代 ， 中")


# ### Generate Sentence

# In[34]:


import random

def generate_text(tokens, N=25):    
    preds = []          
    for i in range(N):   
        learner.model[0].reset()          
        logits, _, _ = learner.model(T(tokens).unsqueeze(1))
        probs = F.softmax(logits).data.cpu().numpy()[-1, :]
        candidates = np.argsort(probs)[::-1]
        while True:
            # Sampling
            candidate = np.random.choice(candidates, p=probs[candidates])
            # Greedy
            # candidate = np.argmax(probs[2:]) + 2
            if candidate > 2:
                print(probs[candidates][:3], probs[candidate])
                preds.append(candidate)
                break
        # for candidate in candidates:
        #     if candidate > 1 and ord(itos[candidate]) > 255 and (random.random() < probs[candidate] or probs[candidate] < 0.2):
        #         print(probs[candidate])
        #         preds.append(candidate)
        #         break
        # tokens  = [preds[-1]]# 
        tokens.append(int(preds[-1]))
        # tokens = [:1]
        print(sp.DecodeIds(tokens)) 
    
generate_text(sp.EncodeAsIds("德国 是 世界 大国 之 一 ， 其 国内 生产 总 值 以 国际 汇率 为主 ， "))


# In[35]:


generate_text(sp.EncodeAsIds("在 现代 印刷 媒体 ， 卡通 是 一种 通常 有 幽默 色 "))


# In[36]:


generate_text(sp.EncodeAsIds("日本 后来 成为 第二次 世界大战 的 轴心国 之一 ， 对 中国 与 南洋 发动 全面 的 战争"))


# In[37]:


generate_text(sp.EncodeAsIds("特朗普 政府 以为 加征 关税 会 令 中国 屈服 ， 这种 策略 肯定 会 适得其反 ， 如果 就业 和 财富"))

