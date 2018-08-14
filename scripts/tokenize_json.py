import gzip
import json
import re
from collections import Counter

from tqdm import tqdm
from opencc import OpenCC
import joblib
import numpy as np

DATAPATH = "/mnt/Intel/zhwiki.json.gz"

CC = OpenCC('t2s')
MIN_FREQ = 500
VOC_SIZE = 10000
PAD = 0
UNK = 1


def clean_text(text):
    text = CC.convert(text)
    text = re.sub(r"'''?", "", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"（.*）", "", text)
    text = re.sub(r"《》", "", text)
    text = re.sub(r"link=\w+\s", " ", text)
    # Remove english
    text = re.sub(r"\b[a-zA-Z\s\.\,\(\)0-9\-]+[\b。]", "", text)
    text = re.sub(" {2,}", " ", text)
    return text


def main():
    cnt = Counter()
    with gzip.open(DATAPATH) as f:
        for i, line in tqdm(enumerate(f.readlines())):
            article = json.loads(line)
            section = article["section_texts"][0]
            section = clean_text(section)
            # print(article["title"], section)
            cnt.update(section)
            # if i == 10000:
            #     break
    print(cnt.most_common(100))
    joblib.dump(cnt, "../data/freq.pkl")
    mapping = {
        char: token + 2 for token, (char, freq) in enumerate(cnt.most_common(VOC_SIZE))
        if freq > MIN_FREQ
    }
    print("Vocab:", len(mapping))
    joblib.dump(mapping, "../data/mapping.pkl")
    texts = []
    with gzip.open(DATAPATH) as f:
        for i, line in tqdm(enumerate(f.readlines())):
            article = json.loads(line)
            section = article["section_texts"][0]
            section = clean_text(section)
            texts.append(
                np.array(list(map(lambda x: mapping.get(x, UNK), section))))
            # if i == 10000:
            #     break
    joblib.dump(np.array(texts), "../data/tokens.pkl")


if __name__ == "__main__":
    main()
