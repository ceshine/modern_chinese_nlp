import subprocess
import re

import sentencepiece as spm
import pandas as pd
from tqdm import tqdm

DATAPATH = "data/ratings.csv"
TMPPATH = "/tmp/ratings.txt"
TMPPATH_WORD = "/tmp/ratings_word.txt"
TARGETPATH = "data/ratings_word.csv"
MODEL_PREFIX = "data/rating_{algorithm}_model"

VOC_SIZE = 7500
PAD = 0
UNK = 1


def main():
    ratings = pd.read_csv(DATAPATH)
    with open(TMPPATH, "w") as fw:
        for row in tqdm(ratings["comment"]):
            fw.write(row + "\n")

    # Tokenization
    res = subprocess.run([
        "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
        "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
    ], stdout=subprocess.PIPE)
    print(res)

    comments = []
    with open(TMPPATH_WORD) as f:
        for line in tqdm(f.readlines()):
            line = re.sub(r"\s+", " ", line).strip()
            comments.append(line)

    # Train BPE Model
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} '
        '--input_sentence_size=20000000 '
        '--character_coverage=0.995 --model_type=bpe'.format(
            TMPPATH_WORD, MODEL_PREFIX.format(algorithm="bpe"), VOC_SIZE
        )
    )

    # Train Unigram Model
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} '
        '--input_sentence_size=20000000 '
        '--character_coverage=0.995 --model_type=unigram'.format(
            TMPPATH_WORD, MODEL_PREFIX.format(algorithm="unigram"), VOC_SIZE
        )
    )

    final_ratings = pd.DataFrame({
        "comment": comments,
        "rating": ratings.rating
    })
    final_ratings.to_csv(TARGETPATH, index=False)


if __name__ == "__main__":
    main()
