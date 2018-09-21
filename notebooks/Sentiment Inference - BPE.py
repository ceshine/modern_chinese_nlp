
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-p torch,joblib,numpy,eli5,jieba,sentencepiece -m')


# In[2]:


from pathlib import Path
import re

import torch
import numpy as np
import joblib
import sentencepiece as spm
import jieba


# In[3]:


path = Path("../data/cache/lm_bpe_douban/")
model = torch.load(path / "sentiment_model.pth")
sp = spm.SentencePieceProcessor()
sp.Load("../data/bpe_model.model")


# In[4]:


model = model.cpu()
model.reset()
model = model.eval()


# In[5]:


UNK = 0
BEG = 1
def get_prediction(texts):
    input_tensor = torch.from_numpy(np.array([1] + sp.EncodeAsIds(texts))).long().unsqueeze(1)
    return model(input_tensor)[0].data.cpu().numpy()[0, 0]


# In[6]:


get_prediction("看了快一半了才发现是mini的广告")


# ## ELI5 (LIME)

# In[7]:


UNK = 2
def get_proba(rows):
    """eli5 only supports classifier, so we need to do some transformations."""
    probs = []
    for texts in rows:
        model.reset()
        texts = re.sub(r"\s+", "", texts)
        texts.replace("qqq", " ")
        input_tensor = torch.from_numpy(
            np.array([1] + sp.EncodeAsIds(texts))
        ).long().unsqueeze(1)
        pos = (np.clip(model(input_tensor)[0].data.cpu().numpy()[0, 0], 1, 5) - 1) / 4
        probs.append(np.array([1-pos, pos]))
    return np.stack(probs, axis=0)


# ### Example 1

# In[8]:


get_proba(["看了快一半了才发现是mini的广告"])


# In[9]:


from eli5.lime import TextExplainer

te = TextExplainer(random_state=42, n_samples=5000)
te.fit(" ".join(jieba.cut("看了快一半了才发现是mini的广告", cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[10]:


te.metrics_


# In[11]:


te.samples_[:10]


# #### Character-based Whitebox

# In[12]:


te = TextExplainer(random_state=42, n_samples=5000, char_based=True)
te.fit("看了快一半了才发现是mini的广告", get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[13]:


te.metrics_


# In[14]:


te.samples_[:10]


# ### Example 2

# In[15]:


te = TextExplainer(random_state=42, n_samples=2000)
te.fit(" ".join(jieba.cut("妈蛋，简直太好看了。最后的 DJqqqbattle 部分，兴奋的我，简直想从座位上站起来一起扭", cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[16]:


te.samples_[:10]


# In[17]:


te.metrics_


# ### Example 3

# In[18]:


te = TextExplainer(random_state=42, n_samples=5000)
te.fit(" ".join(jieba.cut("十亿元很难花掉吗？投资一部《阿修罗》，瞬间就赔光了啊。这设定太没说服力了。", cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[19]:


get_proba([" ".join(jieba.cut("十亿元很难花掉吗？投资一部《阿修罗》，瞬间就赔光了啊。这设定太没说服力了。", cut_all=False))])


# In[20]:


te.metrics_


# ### Example 4

# In[21]:


texts = (
    "爱上了让雷诺。爱上了这个意大利杀手的温情。爱上了他脸部轮廓，和刚毅的线条。"
    "“人生诸多辛苦，是不是只有童年如此。”玛蒂尔达问。里昂说，“一直如此。”这样的话，击中了内心深处。"
)
te = TextExplainer(random_state=42, n_samples=5000)
te.fit(" ".join(jieba.cut(texts, cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[22]:


get_proba([" ".join(jieba.cut(texts, cut_all=False))])


# In[23]:


te.metrics_


# ### Example 5

# In[24]:


texts = "看到泪飙，世上最温情的杀手。 幸运的姑娘在12岁就遇到了真正的男人。"
te = TextExplainer(random_state=42, n_samples=5000)
te.fit(" ".join(jieba.cut(texts, cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[25]:


get_proba([" ".join(jieba.cut(texts, cut_all=False))])


# In[26]:


te.metrics_


# ### Example 6

# In[27]:


texts = (
    "当年的奥斯卡颁奖礼上，被如日中天的《阿甘正传》掩盖了它的光彩，而随着时间的推移，"
    "这部电影在越来越多的人们心中的地位已超越了《阿甘》。每当现实令我疲惫得产生无力感，"
    "翻出这张碟，就重获力量。毫无疑问，本片位列男人必看的电影前三名！回顾那一段经典台词："
    "“有的人的羽翼是如此光辉，即使世界上最黑暗的牢狱，也无法长久地将他围困！”"
)
te = TextExplainer(random_state=42, n_samples=5000)
te.fit(" ".join(jieba.cut(texts, cut_all=False)), get_proba)
te.show_prediction(target_names=["neg", "pos"])


# In[28]:


get_proba([" ".join(jieba.cut(texts, cut_all=False))])


# In[29]:


te.metrics_

