import gzip
import json
import subprocess
import sys

import sentencepiece as spm
import joblib
import numpy as np
from tqdm import tqdm
from opencc import OpenCC

from wiki_tokenize_json import clean_text, filter_texts, SECTION_BLACKLIST

DATAPATH = "/mnt/Intel/zhwiki.json.gz"
TMPPATH = "/mnt/Intel/tmp_texts.txt"
TMPPATH_WORD = "/mnt/Intel/tmp_words.txt"
MODEL_PREFIX = "data/bpe_model"

CC = OpenCC('t2s')
VOC_SIZE = 7500
PAD = 1
UNK = 0


def json_to_txt():
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
                    for paragraph in [x for x in section.split("\n") if len(x) > 50]:
                        paragraph = clean_text(paragraph)
                        if len(paragraph) < 200 or filter_texts(paragraph):
                            continue
                        for sentence in [x for x in paragraph.split("。") if len(x) > 10]:
                            fw.write(sentence + "。\n")


def fit_model(word_seg=True):
    json_to_txt()

    if word_seg:
        res = subprocess.run([
            "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
            "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
        ], stdout=subprocess.PIPE)
        print(res)

    # Train BPE Model
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} '
        '--input_sentence_size=20000000 '
        '--character_coverage=0.995 --model_type=bpe'.format(
            TMPPATH_WORD if word_seg else TMPPATH, MODEL_PREFIX, VOC_SIZE
        )
    )


def tokenize(seg_word):
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PREFIX + ".model")
    tokens = []
    with open(TMPPATH_WORD if seg_word else TMPPATH) as f:
        for _, sentence in tqdm(enumerate(f.readlines())):
            tokens.append(
                np.array(sp.EncodeAsIds(sentence))
            )
    joblib.dump(np.array(tokens), "data/tokens_bpe.pkl")


if __name__ == "__main__":
    seg_word = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "--char":
            seg_word = False
    fit_model(seg_word)
    tokenize(seg_word)
