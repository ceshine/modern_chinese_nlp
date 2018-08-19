
# coding: utf-8

# In[1]:


from pathlib import Path

import torch
import numpy as np
import joblib


# In[2]:


path = Path("../data/cache/lm_douban/")
model = torch.load(path / "sentiment_model.pth")


# In[3]:


model = model.cpu()
model.reset()
model = model.eval()


# In[4]:


mapping = joblib.load("../data/mapping.pkl")


# In[5]:


UNK = 2
def get_prediction(texts):
    input_tensor = torch.from_numpy(np.array([1] + [mapping.get(x, UNK-1) + 1 for x in texts])).long().unsqueeze(1)
    return model(input_tensor)[0].data.cpu().numpy()[0, 0]


# In[6]:


get_prediction("看了快一半了才发现是mini的广告")

