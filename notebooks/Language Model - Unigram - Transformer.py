
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
from fastai.core import T
from fastai.rnn_reg import EmbeddingDropout
from torch.optim import Adam
import torch.nn as nn
import torch
import torch.nn.functional as F
import sentencepiece as spm

from cnlp.fastai_extended import LanguageModelLoader, LanguageModelData, ShuffledLanguageModelLoader


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


bptt = 100
batch_size = 64
n_tok = int(np.max([np.max(x) for x in tokens]) + 1)
trn_loader = ShuffledLanguageModelLoader(
    np.concatenate(trn_tokens), batch_size, bptt, target_length=90, batch_first=True)
val_loader = ShuffledLanguageModelLoader(
    np.concatenate(val_tokens), batch_size,   bptt, target_length=90, batch_first=True)
tst_loader = ShuffledLanguageModelLoader(
    np.concatenate(tst_tokens), batch_size, bptt, target_length=90, batch_first=True)


# In[8]:


sp = spm.SentencePieceProcessor()
sp.Load("../data/unigram_model.model")


# In[10]:


np.sum([np.sum(x == 2) for x in tokens]) # </s>


# In[11]:


sp.DecodeIds(trn_tokens[0].tolist())


# In[12]:


path = Path("../data/cache/lm_unigram_transformer/")
path.mkdir(parents=True, exist_ok=True)
model_data = LanguageModelData(
    path, pad_idx=2, n_tok=n_tok, trn_dl=trn_loader, val_dl=val_loader, test_dl=tst_loader
)


# In[13]:


n_tok


# ### Transformer Model

# In[13]:


drops = np.array([0.1, 0.1, 0.05, 0, 0.1])
learner = model_data.get_model(
    partial(Adam, betas=(0.8, 0.999)),
    emb_sz=300, n_hid=500, n_layers=3,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
    dropoute=drops[3], dropouth=drops[4], qrnn=False
)


# In[14]:


# Courtesy of https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
pytorch_total_params = sum(p.numel() for p in learner.model.parameters())
pytorch_trainable_params = sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
pytorch_total_params, pytorch_trainable_params


# In[14]:


learner = model_data.get_transformer_model(
    partial(Adam, betas=(0.8, 0.99)),
    max_seq_len=trn_loader.max_possible_seq_len,
    emb_sz=480,
    n_head=12,
    n_layer=6,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1
)


# In[15]:


pytorch_total_params = sum(p.numel() for p in learner.model.parameters())
pytorch_trainable_params = sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
pytorch_total_params, pytorch_trainable_params


# In[17]:


learner.clip = 10.
learner.lr_find(start_lr=1e-4, end_lr=1e-2, linear=False)
get_ipython().run_line_magic('time', 'learner.sched.plot()')


# In[16]:


lrs = 1e-3
learner.clip = 20.
get_ipython().run_line_magic('time', 'learner.fit(lrs, 1, wds=0, use_clr=(50, 4), cycle_len=5, use_wd_sched=False)')


# In[17]:


learner.sched.plot_lr()


# In[18]:


learner.save("lm_transformer")
learner.save_encoder("lm_transformer_enc")


# In[19]:


tmp_iter = iter(trn_loader)


# In[20]:


next(tmp_iter)[0].shape


# In[21]:


learner.load("lm_transformer")


# ## Test the model

# In[22]:


learner.model.eval()


# ### Next Character Inference

# In[23]:


tokens = sp.EncodeAsIds("德国 是 世界 大国 之 一 ， 其 国内 生产 总 值 以 国际 汇率 计")
tokens


# In[24]:


iterator = iter(tst_loader)
x, y = next(iterator)


# In[25]:


x.shape, y.shape


# In[26]:


logits = learner.model(x.to("cuda"))
logits.shape


# In[31]:


def eval_tensors(x, y):
    logits = learner.model(x.to("cuda"))
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
    print(x.shape, len(preds[0]))
    return pd.DataFrame({
        "orig": [sp.IdToPiece(int(i)) for i in x[0, 10:].numpy()] + [""], 
        "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2],
        "actual": [""] + [sp.IdToPiece(int(i)) for i in y.numpy()]
    })
tmp = eval_tensors(x[:1, :], y[:90])
tmp[:20]


# In[32]:


tmp.iloc[-20:]


# In[34]:


tmp = eval_tensors(x[1:2, :], y[90:180])
tmp[-20:]


# In[68]:


def eval_text(texts):
    tokens = sp.EncodeAsIds(texts)[:100]
    logits = learner.model(T(tokens).unsqueeze(0))
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
    # preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
    print(len(preds[0]))
    return pd.DataFrame({"orig": sp.EncodeAsPieces(texts)[-90:] + [""], 
                  "pred_1": [""] + preds[0][-90:], "pred_2": [""] + preds[1][-90:], "pred_3": [""] + preds[2][-90:]})


# In[63]:


sp.DecodeIds(x[0, :].numpy().tolist())


# In[64]:


tmp = eval_text(sp.DecodeIds(x[6, :].numpy().tolist()))
tmp


# In[69]:


eval_text("特朗普 政府 以为 加征 关税 会 令 中国 屈服 ， 这种 策略 肯定 会 适得其反 ， 如果 就业 和 财富")


# In[70]:


eval_text("对 中国 与 南洋 发动 全面 的 战争 。 1990 年代 ， 中")

