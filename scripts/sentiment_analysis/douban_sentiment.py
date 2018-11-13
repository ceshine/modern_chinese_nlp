from pathlib import Path
from functools import partial

import joblib
import torch
import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from helperbot import setup_differential_learning_rates, freeze_layers
from helperbot.lr_scheduler import TriangularLR

from dekisugi.dataset import TextDataset
from dekisugi.sequence_model import get_sequence_model, SequenceRegressorBot
from dekisugi.sampler import SortishSampler, SortSampler
from dekisugi.dataloader import DataLoader

UNK = 0
BEG = 1
EMB_DIM = 500

MODEL_PATH = Path("data/cache/douban_dk/")
DEVICE = "cuda:0"


def truncate_tokens(tokens, max_len=100):
    return np.array([
        x[:max_len] for x in tokens
    ])


def filter_entries(tokens, df_ratings, min_len=1, max_len=1000):
    lengths = np.array([len(tokens[i]) for i in range(tokens.shape[0])])
    flags = (lengths >= min_len) & (lengths <= max_len)
    return (
        tokens[flags],
        df_ratings.loc[flags].copy()
    )


def prepare_dataset():
    cache_path = Path("data/cache/douban_sentiment_tokens.pkl")
    if cache_path.exists():
        tokens, df_ratings = joblib.load(cache_path)
    else:
        sp = spm.SentencePieceProcessor()
        sp.Load("data/rating_unigram_model.model")
        df_ratings = pd.read_csv("data/ratings_word.csv")
        tokens = []
        for _, row in tqdm(df_ratings.iterrows(), total=df_ratings.shape[0]):
            tokens.append(sp.EncodeAsIds(row["comment"]))
        assert len(tokens) == df_ratings.shape[0]
        tokens, df_ratings = filter_entries(
            np.array(tokens), df_ratings, min_len=1)
        tokens = truncate_tokens(tokens, max_len=100)
        joblib.dump([tokens, df_ratings], cache_path)
    # df_ratings["rating"] = (df_ratings["rating"] - 1).astype("float32")
    # df_ratings["rating"] = df_ratings["rating"].astype("float32")
    df_ratings["rating"] = ((df_ratings["rating"] - 3) / 2).astype("float32")
    # Split the dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=888)
    train_idx, test_idx = next(sss.split(df_ratings, df_ratings.rating))
    tokens_train, tokens_test = tokens[train_idx], tokens[test_idx]
    y_train = df_ratings.iloc[train_idx][["rating"]].copy().values
    y_test = df_ratings.iloc[test_idx][["rating"]].copy().values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=888)
    val_idx, test_idx = next(sss.split(y_test, y_test))
    tokens_valid, tokens_test = tokens_test[val_idx], tokens_test[test_idx]
    y_valid, y_test = y_test[val_idx], y_test[test_idx]
    del df_ratings
    trn_ds = TextDataset(tokens_train, y_train)
    val_ds = TextDataset(tokens_valid, y_valid)
    tst_ds = TextDataset(tokens_test, y_test)
    print(len(trn_ds), len(val_ds), len(tst_ds))
    return trn_ds, val_ds, tst_ds


def regressor():
    batch_size = 32
    trn_ds, val_ds, tst_ds = prepare_dataset()
    model = get_sequence_model(
        7500,
        emb_sz=500,
        pad_idx=2,
        dropoute=0,
        rnn_hid=500,
        rnn_layers=3,
        bidir=False,
        dropouth=0.2,
        dropouti=0.2,
        wdrop=0.05,
        qrnn=False,
        fcn_layers=[50, 1],
        fcn_dropouts=[0.1, 0.1]
    )

    model = model.to(DEVICE)

    trn_samp = SortishSampler(
        trn_ds.x, key=lambda x: len(trn_ds.x[x]), bs=batch_size)
    val_samp = SortSampler(
        val_ds.x, key=lambda x: len(val_ds.x[x]))
    trn_loader = DataLoader(
        trn_ds, batch_size, transpose=True,
        num_workers=1, pad_idx=2, sampler=trn_samp)
    val_loader = DataLoader(
        val_ds, batch_size * 2, transpose=True,
        num_workers=1, pad_idx=2, sampler=val_samp)

    optimizer_constructor = partial(
        torch.optim.Adam, betas=(0.7, 0.99))
    optimizer = setup_differential_learning_rates(
        optimizer_constructor,
        model,
        (np.array([2**-4, 2**-3, 2**-2, 2**-1, 1]) * 2e-4).tolist()
    )
    bot = SequenceRegressorBot(
        model, trn_loader, val_loader,
        optimizer=optimizer,
        clip_grad=25.,
        log_dir=MODEL_PATH / "logs_reg",
        checkpoint_dir=MODEL_PATH,
        echo=True,
        use_tensorboard=False,
        avg_window=len(trn_loader) // 10 * 2,
        device=DEVICE
    )
    bot.load_encoder(
        prefix="lstm_500x3_emb_7500x500_")
    bot.logger.info(str(model))

    # Train only the last group
    freeze_layers(model.get_layer_groups(), [True] * 4 + [False])
    bot.count_model_parameters()
    n_steps = len(trn_loader) * 1
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        snapshot_interval=len(trn_loader) // 10 * 5,
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=8, ratio=2,
            steps_per_cycle=n_steps)
    )
    bot.remove_checkpoints(keep=1)
    bot.best_performers = []

    # Train the last group and the first grouop
    freeze_layers(model.get_layer_groups(), [False] + [True] * 3 + [False])
    bot.count_model_parameters()
    n_steps = len(trn_loader) * 2
    bot.step = 0
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        snapshot_interval=len(trn_loader) // 10 * 5,
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=8, ratio=2,
            steps_per_cycle=n_steps)
    )
    bot.remove_checkpoints(keep=1)
    bot.best_performers = []

    # Train all groups
    freeze_layers(model.get_layer_groups(), [False] * 5)
    bot.count_model_parameters()
    n_steps = len(trn_loader) * 10
    bot.step = 0
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        snapshot_interval=len(trn_loader) // 10 * 5,
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=64, ratio=8,
            steps_per_cycle=n_steps)
    )
    bot.remove_checkpoints(keep=1)

    tst_samp = SortSampler(
        tst_ds.x, key=lambda x: len(tst_ds.x[x]))
    tst_loader = DataLoader(
        tst_ds, batch_size * 2, transpose=True,
        num_workers=1, pad_idx=2, sampler=tst_samp)
    test_loss = bot.eval(tst_loader)
    bot.logger.info("Test loss: %.4f", test_loss)


def regressor_from_scratch():
    pass


if __name__ == "__main__":
    regressor()
