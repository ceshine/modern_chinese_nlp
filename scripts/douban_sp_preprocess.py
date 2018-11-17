import subprocess
import re

import sentencepiece as spm
import pandas as pd
import click
from tqdm import tqdm

DATAPATH = "data/douban_ratings.csv"
TMPPATH = "/tmp/ratings.txt"
TMPPATH_WORD = "/tmp/ratings_word.txt"
TARGETPATH = "data/ratings_prepared_{seg_word}.csv"
MODEL_PREFIX = "data/rating_{algorithm}_{seg_word}"

VOC_SIZE = 7500


@click.command()
@click.option("--word", is_flag=True)
@click.option("--bpe/--unigram", default=True)
def main(word, bpe):
    seg_word = True if word else False
    algorithm = "bpe" if bpe else "unigram"
    ratings = pd.read_csv(DATAPATH)
    with open(TMPPATH, "w") as fw:
        for row in tqdm(ratings["comment"]):
            fw.write(row + "\n")

    # Segmentation
    if seg_word:
        res = subprocess.run([
            "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
            "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
        ], stdout=subprocess.PIPE)
        print(res)

    comments = []
    with open(TMPPATH_WORD if seg_word else TMPPATH) as f:
        for line in tqdm(f.readlines()):
            line = re.sub(r"\s+", " ", line).strip()
            comments.append(line)

    # Train SentencePiece Model
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} '
        '--input_sentence_size=20000000 '
        '--character_coverage=0.995 --model_type={}'.format(
            TMPPATH_WORD if seg_word else TMPPATH,
            MODEL_PREFIX.format(algorithm=algorithm, seg_word=seg_word),
            VOC_SIZE, algorithm
        )
    )

    final_ratings = pd.DataFrame({
        "comment": comments,
        "rating": ratings.rating
    })
    final_ratings.to_csv(TARGETPATH.format(seg_word=seg_word), index=False)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
