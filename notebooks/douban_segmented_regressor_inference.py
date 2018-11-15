#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dekisugi.sequence_model import get_sequence_model

model = get_sequence_model(
        7500,
        emb_sz=500,
        pad_idx=2,
        dropoute=0,
        rnn_hid=500,
        rnn_layers=3,
        bidir=False,
        dropouth=0.2,
        dropouti=0.2,
        wdrop=0.05,
        qrnn=False,
        fcn_layers=[50, 1],
        fcn_dropouts=[0.1, 0.1]
    )


# In[2]:


import torch
model.load_state_dict(torch.load("../data/cache/douban_dk/snapshot_seq_regressor_0.164828.pth"))


# In[3]:


import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("../data/rating_unigram_model.model")


# In[4]:


import numpy as np

def restore_rating(scores):
    return scores * 2 + 3

model.cpu()
def get_prediction(texts):
    model.eval()
    input_tensor = torch.LongTensor(sp.EncodeAsIds(texts)).unsqueeze(1)
    return restore_rating(model(input_tensor).data.cpu().numpy()[0, 0])


# In[5]:


get_prediction("看 了 快 一半 了 才 发现 是 mini 的 广告")


# In[6]:


get_prediction("妈蛋 ， 简直 太 好看 了 。 最后 的 DJ battle 部分 ， 兴奋 的 我 ， 简直 想 从 座位 上 站 起来 一起 扭")


# In[7]:


get_prediction("太 烂 了 ， 难看 至极 。")


# In[8]:


get_prediction("看完 之后 很 生气 ！ 剧情 太差 了")


# ## For Debug Purpose

# ### From the CSV

# In[9]:


import pandas as pd
df_ratings = pd.read_csv("../data/ratings_word.csv")


# In[10]:


def evaluate_labeled_row(idx):
    return (
        get_prediction(df_ratings.iloc[idx]["comment"]),
        df_ratings.iloc[idx]["rating"], 
        df_ratings.iloc[idx]["comment"])


# In[11]:


evaluate_labeled_row(1024)


# ### From the Tokenized Dump. Recalculate the Validation Score.

# In[12]:


import joblib
tokens, df_ratings = joblib.load("../data/cache/douban_sentiment_tokens.pkl")
df_ratings["rating"] = ((df_ratings["rating"] - 3) / 2).astype("float32")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=888)
train_idx, test_idx = next(sss.split(df_ratings, df_ratings.rating))
tokens_train, tokens_test = tokens[train_idx], tokens[test_idx]
y_train = df_ratings.iloc[train_idx][["rating"]].copy().values
y_test = df_ratings.iloc[test_idx][["rating"]].copy().values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=888)
val_idx, test_idx = next(sss.split(y_test, y_test))
tokens_valid, tokens_test = tokens_test[val_idx], tokens_test[test_idx]
y_valid, y_test = y_test[val_idx], y_test[test_idx]


# In[14]:


from dekisugi.dataset import TextDataset
from dekisugi.dataloader import DataLoader
from dekisugi.sampler import SortishSampler, SortSampler
trn_ds = TextDataset(tokens_train, y_train)
val_ds = TextDataset(tokens_valid, y_valid)
tst_ds = TextDataset(tokens_test, y_test)
val_samp = SortSampler(
    val_ds.x, key=lambda x: len(val_ds.x[x]))
val_loader = DataLoader(
    val_ds, 128, transpose=True, pin_memory=True,
    num_workers=1, pad_idx=2, sampler=val_samp)


# In[15]:


model.cuda()
model.eval()
losses = []
with torch.set_grad_enabled(False):
    for input_tensor, target_tensor in val_loader:
        losses.append(np.square(model(input_tensor)[:, 0] - target_tensor[:, 0]).data.numpy())


# In[16]:


np.mean(np.concatenate(losses))


# In[17]:


sp.DecodeIds(val_ds[20][0].tolist())


# #### UseThe  Previously Used Metric

# In[18]:


model.cuda()
model.eval()
losses = []
with torch.set_grad_enabled(False):
    for input_tensor, target_tensor in val_loader:
        losses.append(np.square(
            2 * (model(input_tensor)[:, 0] - target_tensor[:, 0])).data.numpy())


# In[19]:


np.mean(np.concatenate(losses))


# In[ ]:




