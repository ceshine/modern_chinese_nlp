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
model.load_state_dict(torch.load("../data/cache/douban_dk_noseg/snapshot_seq_regressor_0.153608.pth"))


# In[3]:


import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("../data/rating_unigram_False.model")


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


get_prediction("看了快一半了才发现是mini的广告")


# In[6]:


get_prediction("妈蛋，简直太好看了。最后的DJ battle部分，兴奋的我，简直想从座位上站起来一起扭")


# In[7]:


get_prediction("太烂了，难看至极。")


# In[8]:


get_prediction("看完之后很生气！剧情太差了")


# ## For Debug Purpose

# ### From the CSV

# In[9]:


import pandas as pd
df_ratings = pd.read_csv("../data/ratings_prepared_False.csv")


# In[10]:


def evaluate_labeled_row(idx):
    return (
        get_prediction(df_ratings.iloc[idx]["comment"]),
        df_ratings.iloc[idx]["rating"], 
        df_ratings.iloc[idx]["comment"])


# In[11]:


evaluate_labeled_row(1024)


# ### Reproduce the dataset, and Recalculate the Validation and Test Scores

# In[12]:


import tqdm
WORD_SEG = False
sp = spm.SentencePieceProcessor()
sp.Load(f"../data/rating_unigram_{WORD_SEG}.model")
df_ratings = pd.read_csv(f"../data/ratings_prepared_{WORD_SEG}.csv")
df_ratings["rating"] = ((df_ratings["rating"] - 3) / 2).astype("float32")
tokens = []
for _, row in tqdm.tqdm_notebook(df_ratings.iterrows(), total=df_ratings.shape[0]):
    tokens.append(sp.EncodeAsIds(row["comment"]))
assert len(tokens) == df_ratings.shape[0]


# In[13]:


def filter_entries(tokens, df_ratings, min_len=1, max_len=1000):
    lengths = np.array([len(tokens[i]) for i in range(tokens.shape[0])])
    flags = (lengths >= min_len) & (lengths <= max_len)
    return (
        tokens[flags],
        df_ratings.loc[flags].copy()
    )
def truncate_tokens(tokens, max_len=100):
    return np.array([
        x[:max_len] for x in tokens
    ])
tokens, df_ratings = filter_entries(
    np.array(tokens), df_ratings, min_len=1)
tokens = truncate_tokens(tokens, max_len=100)


# In[14]:


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


# In[15]:


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


# #### Validation

# In[16]:


model.cuda()
model.eval()
losses = []
with torch.set_grad_enabled(False):
    for input_tensor, target_tensor in val_loader:
        losses.append(np.square(model(input_tensor)[:, 0] - target_tensor[:, 0]).data.numpy())


# In[17]:


np.mean(np.concatenate(losses))


# In[18]:


sp.DecodeIds(val_ds[21][0].tolist())


# #### Use the  Previously Used Metric (Validation)

# In[19]:


model.cuda()
model.eval()
losses = []
with torch.set_grad_enabled(False):
    for input_tensor, target_tensor in tqdm.tqdm_notebook(val_loader):
        losses.append(np.square(
            2 * (model(input_tensor)[:, 0] - target_tensor[:, 0])).data.numpy())


# In[20]:


np.mean(np.concatenate(losses))


# #### Test

# In[21]:


tst_samp = SortSampler(
        tst_ds.x, key=lambda x: len(tst_ds.x[x]))
tst_loader = DataLoader(
    tst_ds, 128, transpose=True,
    num_workers=1, pad_idx=2, sampler=tst_samp)


# In[22]:


model.cuda()
model.eval()
losses = []
with torch.set_grad_enabled(False):
    for input_tensor, target_tensor in tqdm.tqdm_notebook(tst_loader):
        losses.append(np.square(model(input_tensor)[:, 0] - target_tensor[:, 0]).data.numpy())


# In[23]:


np.mean(np.concatenate(losses))


# In[ ]:




