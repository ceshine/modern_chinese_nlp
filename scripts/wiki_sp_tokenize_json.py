"""SentencePiece Tokenization for Wiki Dataset

Example:
  * python scripts/wiki_sp_tokenize_json.py --word --unigram
"""
import gzip
import json
import subprocess
from pathlib import Path

import sentencepiece as spm
import joblib
import numpy as np
import click
from tqdm import tqdm
from opencc import OpenCC

from wiki_tokenize_json import clean_text, filter_texts, SECTION_BLACKLIST

DATAPATH = "/mnt/Intel/zhwiki.json.gz"
TMPPATH = "/mnt/Intel/tmp_texts.txt"
TMPPATH_WORD = "/mnt/Intel/tmp_words.txt"
MODEL_PREFIX = "data/{algorithm}_{seg_word}_model"

CC = OpenCC('t2s')
VOC_SIZE = 7500
PAD = 1
UNK = 0


def json_to_txt():
    with gzip.open(DATAPATH) as f:
        with open(TMPPATH, "w") as fw:
            for _, line in tqdm(enumerate(f.readlines())):
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


def fit_model(seg_word=True, algorithm="bpe"):
    if not Path(TMPPATH).exists():
        json_to_txt()

    if seg_word:
        print("Performing word segmentation...")
        res = subprocess.run([
            "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
            "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
        ], stdout=subprocess.PIPE)
        print(res)

    # Train Model
    print("Training model...")
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} '
        '--input_sentence_size=20000000 '
        '--character_coverage=0.995 --model_type={algorithm}'.format(
            TMPPATH_WORD if seg_word else TMPPATH,
            MODEL_PREFIX.format(algorithm=algorithm, seg_word=seg_word),
            VOC_SIZE, algorithm="unigram"
        )
    )


def tokenize(seg_word=True, algorithm="bpe"):
    print("Tokenizing...")
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PREFIX.format(
        algorithm=algorithm, seg_word=seg_word) + ".model")
    tokens = []
    with open(TMPPATH_WORD if seg_word else TMPPATH) as f:
        for _, sentence in tqdm(enumerate(f.readlines())):
            tokens.append(
                np.array(sp.EncodeAsIds(sentence))
            )
    joblib.dump(np.array(tokens), f"data/tokens_{algorithm}_{seg_word}.pkl")


@click.command()
@click.option("--word", is_flag=True)
@click.option("--bpe/--unigram", default=True)
def main(word, bpe):
    seg_word = True if word else False
    algorithm = "bpe" if bpe else "unigram"
    # fit_model(seg_word, algorithm)
    tokenize(seg_word, algorithm)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
