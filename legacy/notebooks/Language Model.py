
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")


# In[1]:


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


# In[3]:


tokens = joblib.load("../data/tokens.pkl")


# In[4]:


# Filter out empty texts
tokens = [x for x in tokens if x.shape[0] > 0]


# In[5]:


trn_tokens, val_tokens = train_test_split(tokens, test_size=0.2)
val_tokens, tst_tokens = train_test_split(val_tokens, test_size=0.5)


# In[6]:


def get_voc_stats(tokens):
    total_tokens = np.sum([x.shape[0] for x in tokens])
    unks = np.sum([np.sum(x == 1) for x in tokens])
    print("Total tokens: %d\nUnknown Percentage: %.2f %%" % (total_tokens, unks * 100 / total_tokens))
get_voc_stats(tokens)


# In[7]:


bptt = 100
batch_size = 128
n_tok = int(np.max([np.max(x) for x in tokens]) + 1)
trn_loader = LanguageModelLoader(
    np.concatenate(trn_tokens), batch_size, bptt)
val_loader = LanguageModelLoader(
    np.concatenate(val_tokens), batch_size, bptt)
tst_loader = LanguageModelLoader(
    np.concatenate(tst_tokens), batch_size, bptt)


# In[8]:


from collections import Counter
tmp = []
for i in range(10000):
    for j in range(1, trn_tokens[i].shape[0]):
        if trn_tokens[i][j] == 1:
            tmp.append(trn_tokens[i][j-1])
Counter(tmp).most_common(10)


# In[9]:


from collections import Counter
tmp = []
for i in range(10000):
    for j in range(1, trn_tokens[i].shape[0]-1):
        if trn_tokens[i][j] == 4:
            tmp.append(trn_tokens[i][j+1])
Counter(tmp).most_common(10)


# In[10]:


mapping = joblib.load("../data/mapping.pkl")


# In[11]:


itos = ['<unk>'] *  n_tok
for k, v in mapping.items():
    itos[v] = k


# In[12]:


itos[10]


# In[13]:


path = Path("../data/cache/lm/")
path.mkdir(parents=True, exist_ok=True)
model_data = LanguageModelData(
    path, pad_idx=0, n_tok=n_tok, trn_dl=trn_loader, val_dl=val_loader, test_dl=tst_loader
)


# ### QRNN Model

# In[21]:


drops = np.array([0.05, 0.1, 0.05, 0, 0.1])
learner = model_data.get_model(
    partial(Adam, betas=(0.8, 0.999)),
    emb_sz=300, n_hid=500, n_layers=4,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
    dropoute=drops[3], dropouth=drops[4], qrnn=True
)


# In[20]:


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

# In[45]:


drops = np.array([0.1, 0.1, 0.05, 0, 0.1])
learner = model_data.get_model(
    partial(Adam, betas=(0.8, 0.999)),
    emb_sz=300, n_hid=500, n_layers=3,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2],
    dropoute=drops[3], dropouth=drops[4], qrnn=False
)


# In[10]:


learner.clip = 25.
learner.lr_find(start_lr=1e-5, end_lr=1, linear=False)
learner.sched.plot()


# In[46]:


lrs = 6e-3
learner.fit(lrs, 1, wds=1e-7, use_clr=(50, 3), cycle_len=10, use_wd_sched=True)


# In[47]:


lrs = 1e-3
learner.fit(lrs, 1, wds=1e-7, use_clr=(50, 3), cycle_len=10, use_wd_sched=True)


# In[48]:


learner.sched.plot_lr()


# In[49]:


learner.save("lm_lstm")
learner.save_encoder("lm_lstm_enc")


# In[68]:


# Use tons of memory...
# pred, targ = learner.predict_with_targs(is_test=True)


# In[33]:


tmp_iter = iter(trn_loader)


# In[65]:


next(tmp_iter)[0].shape


# In[66]:


learner.load("lm_lstm")


# ## Test the model

# In[50]:


learner.model.eval()


# ### Next Character Inference

# In[51]:


tokens = list(map(lambda x: mapping.get(x, 1), "德国是世界大国之一，其国内生产总值以国际汇率计"))
tokens


# In[52]:


logits, _, _ = learner.model(T(tokens).unsqueeze(1))
logits.shape


# In[53]:


sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
preds = []
for i in range(1, 4):
      preds.append(list(map(lambda x: itos[x], sorted_idx[:, -i])))
# preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
pd.DataFrame({"orig": [x for x in "德国是世界大国之一，其国内生产总值以国际汇率计 "], 
              "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2]})


# In[54]:


def eval(texts):
    learner.model[0].reset()
    tokens = list(map(lambda x: mapping.get(x, 1), texts))
    logits, _, _ = learner.model(T(tokens).unsqueeze(1))
    sorted_idx = np.argsort(logits.data.cpu().numpy(), 1)
    preds = []
    for i in range(1, 4):
          preds.append(list(map(lambda x: itos[x], sorted_idx[:, -i])))
    # preds = list(map(lambda x: itos[x], np.argmax(logits.data.cpu().numpy(), 1)))
    return pd.DataFrame({"orig": [x for x in texts + " "], 
                  "pred_1": [""] + preds[0], "pred_2": [""] + preds[1], "pred_3": [""] + preds[2]})


# In[55]:


eval("在现代印刷媒体，卡通是一种通常有幽默色")


# In[56]:


eval("对中国与南洋发动全面的战争。\n1990年代，中")


# ### Generate Sentence

# In[1]:


import random

def get_tokens(texts):
    return list(map(lambda x: mapping.get(x, 1), texts))

def generate_text(tokens,N=25):    
    preds = []          
    for i in range(N):   
        learner.model[0].reset()          
        logits, _, _ = learner.model(T(tokens).unsqueeze(1))
        probs = F.softmax(logits).data.cpu().numpy()[-1, :]
        candidates = np.argsort(probs)[::-1]
        while True:
            candidate = np.random.choice(candidates, p=probs[candidates])
            if candidate > 1 and ord(itos[candidate]) > 255:
                print(probs[candidates][:3], probs[candidate])
                preds.append(candidate)
                break
        # for candidate in candidates:
        #     if candidate > 1 and ord(itos[candidate]) > 255 and (random.random() < probs[candidate] or probs[candidate] < 0.2):
        #         print(probs[candidate])
        #         preds.append(candidate)
        #         break
        # tokens  = [preds[-1]]# 
        tokens.append(preds[-1])
        # tokens = [:1]
        print("".join([itos[x] for x in tokens])) 
    
generate_text(get_tokens("德国是世界大国之一，其国内生产总值以国际汇率为主，"))


# In[73]:


generate_text(get_tokens("在现代印刷媒体，卡通是一种通常有幽默色"))


# In[74]:


generate_text(get_tokens("在现代印刷媒体， 第"))


# In[75]:


generate_text(get_tokens("日本后来成为第二次世界大战的轴心国之一，对中国与南洋发动全面的战争。"))           


# In[76]:


generate_text(get_tokens("传说日本于公元前660年2月11日建国，在公元4世纪出现首个统一政权，并于大化改新中确立了天皇的中央集权体制"
                         "。至平安时代结束前，日本透过文字、宗教、艺术、政治制度等从汉文化引进的事物，开始派生出今日为人所知的文化基"
                         "础。12世纪后的六百年间，日本由武家阶级创建的数个幕府及军事强人政权实际掌权，期间包括了政治纷乱的南北朝与"
                         "战国"))           


# In[77]:


generate_text(get_tokens("特朗普政府以为加征关税会令中国屈服，这种策略肯定会适得其反。如果就业和财富"))


# In[78]:


generate_text(get_tokens("香港有半数人住在公屋，如今这里意外成为Instagram上备受欢迎的拍照地"))


# In[79]:


generate_text(get_tokens("香港有半数人住在公屋，如今这里意外成为Instagram上备受欢迎的拍照地，"
                         "呈现出一个与天际线中的香港不同的景象"))


# In[65]:


generate_text(get_tokens("香港有半数人住在公屋，如今这里意外成为Insta"))

