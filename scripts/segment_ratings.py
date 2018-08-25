import subprocess
import re

import pandas as pd
from tqdm import tqdm

DATAPATH = "data/ratings.csv"
TMPPATH = "/tmp/ratings.txt"
TMPPATH_WORD = "/tmp/ratings_word.txt"
TARGETPATH = "data/ratings_word.csv"


def main():
    ratings = pd.read_csv(DATAPATH)
    with open(TMPPATH, "w") as fw:
        for row in ratings["comment"]:
            fw.write(row + "\n")
    # Tokenization
    res = subprocess.run([
        "thulac", "-model_dir", "/mnt/SSD_Data/openai_nlp/THULAC/models/",
        "-seg_only", "-input", TMPPATH, "-output", TMPPATH_WORD
    ], stdout=subprocess.PIPE)
    print(res)

    comments = []
    with open(TMPPATH_WORD) as f:
        for line in f.readlines():
            line = re.sub(r"\s+", " ", line).strip()
            comments.append(line)

    final_ratings = pd.DataFrame({
        "comment": comments,
        "rating": ratings.rating
    })
    final_ratings.to_csv(TARGETPATH, index=False)


if __name__ == "__main__":
    main()
