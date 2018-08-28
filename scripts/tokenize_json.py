import gzip
import subprocess
import json
import re
import sys
import pathlib
from collections import Counter

from tqdm import tqdm
from opencc import OpenCC
import joblib
import numpy as np

DATAPATH = "/mnt/Intel/zhwiki.json.gz"
TMPPATH = "/mnt/Intel/tmp_texts.txt"
TMPPATH_WORD = "/mnt/Intel/tmp_words.txt"

CC = OpenCC('t2s')
MIN_FREQ = 500
VOC_SIZE = 10000
MIN_FREQ_WORD = 750
VOC_SIZE_WORD = 30000
PAD = 0
UNK = 1

SECTION_BLACKLIST = [
    "相关条目", "外部链接", "参看",
    "注释", "参考文献", "参考书目",
    "扩展阅读", "延伸阅读", "外部连结",
    "相关著作", "分类", "图片", "扩-{展}-阅读",
    "参考来源"
]


def clean_text(text):
    text = " ".join([x for x in text.split("\n") if len(x) > 50])
    text = CC.convert(text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"（.*）", "", text)
    text = re.sub(r"\(.*\)", "", text)
    text = re.sub(r"\u200B", "", text)
    text = re.sub(r"\-\{.*\}\-", "", text)
    text = re.sub(r"《》", "", text)
    # text = re.sub(r"\-{2,}", "", text)
    text = re.sub(r"link=\w+\s", " ", text)
    text = re.sub(r"File:.+\|", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[^a-zA-Z0-9/\-]) (?=[^a-zA-Z0-9/\-])", "", text)
    text = re.sub(r"(?<=[a-zA-Z0-9/\-]) (?=[^a-zA-Z0-9/\-])", "", text)
    text = re.sub(r"(?<=[^a-zA-Z0-9/\-]) (?=[a-zA-Z0-9/\-])", "", text)
    text = re.sub(r"\*", " ", text)
    text = re.sub(r" $", "", text)
    text = re.sub(r"^ ", "", text)
    # Remove english
    text = re.sub(
        r"(?<=[^a-zA-Z0-9]\s)[a-zA-Z\s\.\,\(\)0-9\-\;\"]+(?=[\s。][^a-zA-Z0-9])", "", text)
    text = re.sub(
        r"(?<=[^a-zA-Z0-9]\s)[a-zA-Z\s\.\,\(\)0-9\-\;\"]$", "", text)
    return text


def filter_texts(texts):
    ascii_cnt = len([1 for x in texts if ord(x) < 256])
    if ascii_cnt / len(texts) > 0.5:
        return True
    return False


def main(char_level):
    """
    * 2-pass for character-level tokenization.
    * 3-pass for word-level tokenization.
    """
    cnt = Counter()
    with gzip.open(DATAPATH) as f:
        with open(TMPPATH, "w") as fw:
            for i, line in tqdm(enumerate(f.readlines())):
                article = json.loads(line)
                if "年表" in article["title"] or "列表" in article["title"]:
                    continue
                for title, section in zip(article["section_titles"], article["section_texts"]):
                    title = CC.convert(title)
                    if title in SECTION_BLACKLIST:
                        continue
                    section = clean_text(section)
                    if len(section) < 200 or filter_texts(section):
                        continue
                    if char_level:
                        cnt.update(section)
                    # fw.write(title + "===\n")
                    fw.write(section + "\n")
    if char_level:
        print(cnt.most_common(100))
        joblib.dump(cnt, "data/freq.pkl")
        mapping = {
            char: token + 2 for token, (char, freq) in enumerate(cnt.most_common(VOC_SIZE))
            if freq > MIN_FREQ
        }
        print("Vocab:", len(mapping))
        joblib.dump(mapping, "data/mapping.pkl")
        texts = []
        with open(TMPPATH) as f:
            for i, section in tqdm(enumerate(f.readlines())):
                texts.append(
                    np.array(list(map(lambda x: mapping.get(x, UNK), section))))
        joblib.dump(np.array(texts), "data/tokens.pkl")
    else:
        # Segmentation
        res = subprocess.run([
            "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
            "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
        ], stdout=subprocess.PIPE)
        print(res)
        # Count tokens
        with open(TMPPATH_WORD) as f:
            for _, section in tqdm(enumerate(f.readlines())):
                cnt.update(re.sub(r"\s+", " ", section).split(" "))
        print(cnt.most_common(100))
        joblib.dump(cnt, "data/freq_word.pkl")
        mapping = {
            word: token + 2 for token, (word, freq) in enumerate(
                cnt.most_common(VOC_SIZE_WORD))
            if freq > MIN_FREQ_WORD
        }
        print("Vocab:", len(mapping))
        joblib.dump(mapping, "data/mapping_word.pkl")
        # The actual tokenization part
        tokens = []
        with open(TMPPATH_WORD) as f:
            for i, section in tqdm(enumerate(f.readlines())):
                tokens.append(
                    np.array(list(map(
                        lambda x: mapping.get(x, UNK),
                        re.sub(r"\s+", " ", section).split(" ")
                    ))))
        joblib.dump(np.array(tokens), "data/tokens_word.pkl")


if __name__ == "__main__":
    char_level = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "--word":
            char_level = False
    main(char_level=char_level)
