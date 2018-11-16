from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import sentencepiece as spm
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helperbot.lr_scheduler import TriangularLR
from helperbot import setup_differential_learning_rates, freeze_layers

from dekisugi.language_model import RNNLanguageModel
from dekisugi.language_model import LanguageModelLoader, get_language_model, LMBot

MODEL_PATH = Path("data/cache/douban_dk_noseg/")

UNK = 0
BEG = 1
EMB_DIM = 500
DEVICE = "cuda:0"
BATCH_SIZE = 64
BPTT = 75


def split_tokens(tokens):
    # Filter out empty texts
    tokens = [x for x in tokens if x.shape[0] > 0]
    # Set shuffle = False to keep sentences from the same paragraph together
    trn_tokens, val_tokens = train_test_split(
        tokens, test_size=0.2, shuffle=False, random_state=898)
    val_tokens, tst_tokens = train_test_split(
        val_tokens, test_size=0.5, shuffle=False, random_state=898)
    voc_sz = int(np.max([np.max(x) for x in tokens]) + 1)
    return trn_tokens, val_tokens, tst_tokens, voc_sz


def prepare_dataset(model: RNNLanguageModel):
    sp = spm.SentencePieceProcessor()
    sp.Load("data/rating_unigram_False.model")
    itos_orig = []
    with open("data/unigram_False_model.vocab", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            itos_orig.append(line.split("\t")[0])
    itos = []
    with open("data/rating_unigram_False.vocab", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            itos.append(line.split("\t")[0])
    mapping_orig = {s: idx for idx, s in enumerate(itos_orig)}
    cache_path = Path("/tmp/douban_tokens.pkl")
    if cache_path.exists():
        tokens = joblib.load("data/cache/douban_tokens.pkl")
    else:
        df_ratings = pd.read_csv("data/ratings_prepared.csv")
        tokens = []
        for i, row in tqdm(df_ratings.iterrows(), total=df_ratings.shape[0]):
            tokens.append(np.array([BEG] + sp.EncodeAsIds(row["comment"])))
        assert len(tokens) == df_ratings.shape[0]
        joblib.dump(tokens, cache_path)
        del df_ratings
    trn_tokens, val_tokens, tst_tokens, voc_sz = split_tokens(tokens)
    # Prepare the embedding matrix
    model.embeddings.voc_sz = voc_sz
    assert model.embeddings.encoder.weight.shape[1] == EMB_DIM
    new_matrix = np.random.uniform(
        -.1, .1, (voc_sz, EMB_DIM)).astype("float32")
    hits = 0
    for i, w in enumerate(itos):
        if w in mapping_orig:
            new_matrix[i] = model.embeddings.encoder.weight[mapping_orig[w]].to(
                "cpu").detach().numpy()
            hits += 1
    new_matrix[BEG, :] = 0
    print("Hit rate of vocabulary: %.2f%%" % (hits * 100 / len(itos)))
    model.embeddings.encoder.weight = nn.Parameter(
        torch.from_numpy(new_matrix).float().to(DEVICE))
    if model.tie_weights:
        model.decoder[-1].weight = model.embeddings.encoder.weight
        assert model.decoder[-1].weight is model.embeddings.encoder.weight
    else:
        new_matrix = np.random.uniform(
            -.1, .1, (voc_sz, EMB_DIM)).astype("float32")
        hits = 0
        for i, w in enumerate(itos):
            if w in mapping_orig:
                new_matrix[i] = model.decoder[-1].weight[mapping_orig[w]].to(
                    "cpu").detach().numpy()
                hits += 1
        new_matrix[BEG, :] = 0
        model.decoder[-1].weight = nn.Parameter(
            torch.from_numpy(new_matrix).float().to(DEVICE))
    return trn_tokens, val_tokens, tst_tokens, voc_sz


def main():
    model = get_language_model(
        7500,
        emb_sz=500,
        pad_idx=2,
        dropoute=0,
        rnn_hid=500,
        rnn_layers=3,
        bidir=False,
        dropouth=0.1,
        dropouti=0.1,
        wdrop=0.1,
        qrnn=False,
        tie_weights=True
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(
        "data/cache/lm_unigram_dk_noseg/snapshot_LM_4.2663.pth"))
    trn_tokens, val_tokens, tst_tokens, voc_sz = prepare_dataset(model)
    trn_loader = LanguageModelLoader(
        np.concatenate(trn_tokens), BATCH_SIZE, BPTT)
    val_loader = LanguageModelLoader(
        np.concatenate(val_tokens), BATCH_SIZE, BPTT, randomize=False)
    tst_loader = LanguageModelLoader(
        np.concatenate(tst_tokens), BATCH_SIZE, BPTT, randomize=False)
    optimizer_constructor = partial(
        torch.optim.Adam, betas=(0.8, 0.999))
    optimizer = setup_differential_learning_rates(
        optimizer_constructor,
        model,
        (np.array([2 ** -3, 2**-2, 2**-1, 1]) * 1e-3).tolist()
    )
    freeze_layers(model.get_layer_groups(), [True] * 3 + [False])
    bot = LMBot(
        model, trn_loader, val_loader,
        optimizer=optimizer,
        clip_grad=25.,
        log_dir=MODEL_PATH / "logs",
        checkpoint_dir=MODEL_PATH,
        echo=True,
        use_tensorboard=False,
        avg_window=len(trn_loader) // 10 * 2,
        device=DEVICE
    )
    bot.logger.info(str(model))
    assert bot.model.embeddings.encoder.weight is bot.model.decoder[-1].weight
    n_steps = len(trn_loader) * 2
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        # eval after finished one epoch to maintain training contexts
        # TODO: cache training context before eval and resotre after eval
        snapshot_interval=len(trn_loader),
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=8, ratio=2,
            steps_per_cycle=n_steps)
    )
    bot.remove_checkpoints(keep=1)
    assert bot.model.embeddings.encoder.weight is bot.model.decoder[-1].weight
    # bot.load_model(bot.best_performers[0][1])
    bot.best_performers = []
    # Unfreeze and continue training
    freeze_layers(model.get_layer_groups(), [False] * 4)
    bot.count_model_parameters()
    n_steps = len(trn_loader) * 10
    bot.step = 0
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        # eval after finished one epoch to maintain training contexts
        # TODO: cache training context before eval and resotre after eval
        snapshot_interval=len(trn_loader),
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=64, ratio=6,
            steps_per_cycle=n_steps)
    )
    bot.remove_checkpoints(keep=1)
    bot.export_encoder(
        bot.best_performers[0][1],
        prefix="lstm_500x3_emb_7500x500_")
    test_loss = bot.eval(tst_loader)
    bot.logger.info("Test loss: %.4f", test_loss)


if __name__ == "__main__":
    main()
