
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

from cnlp.fastai_extended import LanguageModelLoader, LanguageModelData


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


# In[20]:


bptt = 200
batch_size = 128
n_tok = int(np.max([np.max(x) for x in tokens]) + 1)
trn_loader = LanguageModelLoader(
    np.concatenate(trn_tokens), batch_size, bptt, target_length=160, batch_first=True)
val_loader = LanguageModelLoader(
    np.concatenate(val_tokens), batch_size, bptt, target_length=160, batch_first=True)
tst_loader = LanguageModelLoader(
    np.concatenate(tst_tokens), batch_size, bptt, target_length=160, batch_first=True)


# In[9]:


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


# In[36]:


learner = model_data.get_transformer_model(
    partial(Adam, betas=(0.9, 0.999)),
    max_seq_len=trn_loader.max_possible_seq_len,
    emb_sz=300,
    n_head=12,
    n_layer=3,
    embd_pdrop=0.05,
    attn_pdrop=0.1,
    resid_pdrop=0.1
)


# In[37]:


pytorch_total_params = sum(p.numel() for p in learner.model.parameters())
pytorch_trainable_params = sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
pytorch_total_params, pytorch_trainable_params


# In[38]:


learner.clip = 10.
learner.lr_find(start_lr=1e-4, end_lr=1e-2, linear=False)
get_ipython().run_line_magic('time', 'learner.sched.plot()')


# In[39]:


lrs = 2e-3
learner.clip = 20.
get_ipython().run_line_magic('time', 'learner.fit(lrs, 1, wds=0, use_clr=(50, 4), cycle_len=10, use_wd_sched=False)')


# In[40]:


learner.sched.plot_lr()


# In[41]:


learner.save("lm_transformer")
learner.save_encoder("lm_transformer_enc")


# In[42]:


tmp_iter = iter(trn_loader)


# In[43]:


next(tmp_iter)[0].shape


# In[45]:


learner.load("lm_transformer")


# ## Test the model

# In[46]:


learner.model.eval()


# ### Next Character Inference

# In[47]:


tokens = sp.EncodeAsIds("德国 是 世界 大国 之 一 ， 其 国内 生产 总 值 以 国际 汇率 计")
tokens


# In[48]:


iterator = iter(tst_loader)
x, y = next(iterator)


# In[49]:


x.shape, y.shape


# In[50]:


logits = learner.model(x.to("cuda"))
logits.shape


# In[51]:


def eval_tensors(x, y):
    logits = learner.model(x.to("cuda"))
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
    return pd.DataFrame({
        "orig": [sp.IdToPiece(int(i)) for i in x[0, 40:].numpy()] + [""], 
        "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2],
        "actual": [""] + [sp.IdToPiece(int(i)) for i in y.numpy()]
    })
tmp = eval_tensors(x[:1, :], y[:160])
tmp[:20]


# In[52]:


tmp.iloc[-20:]


# In[53]:


tmp = eval_tensors(x[1:2, :], y[160:320])
tmp[-20:]


# In[54]:


def eval_text(texts):
    tokens =sp.EncodeAsIds(texts)
    logits = learner.model(T(tokens).unsqueeze(0))
    print(logits.shape)
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append([sp.IdToPiece(x) for x in sorted_idx[:, -i].tolist()])
    # preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
    return pd.DataFrame({"orig": sp.EncodeAsPieces(texts)[-160:] + [""], 
                  "pred_1": [""] + preds[0][-160:], "pred_2": [""] + preds[1][-160:], "pred_3": [""] + preds[2][-160:]})


# In[55]:


sp.DecodeIds(x[0, :].numpy().tolist())


# In[56]:


tmp = eval_text(sp.DecodeIds(x[6, :].numpy().tolist()))
tmp


# In[57]:


eval_text("特朗普 政府 以为 加征 关税 会 令 中国 屈服 ， 这种 策略 肯定 会 适得其反 ， 如果 就业 和 财富")


# In[58]:


eval_text("对 中国 与 南洋 发动 全面 的 战争 。 1990 年代 ， 中")

