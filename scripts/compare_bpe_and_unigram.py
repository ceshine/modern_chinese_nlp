import joblib
import numpy as np

tokens_bpe = joblib.load("data/tokens_bpe.pkl")
tokens_unigram = joblib.load("data/tokens_unigram.pkl")

n_diff = 0
for bpe, unigram in zip(tokens_bpe, tokens_unigram):
    if np.array_equal(bpe, unigram):
        continue
    n_diff += 1

print(n_diff, tokens_bpe.shape[0], tokens_unigram.shape[0])
