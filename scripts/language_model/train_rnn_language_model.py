from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import torch

from dekisugi.language_model import LanguageModelLoader, get_language_model, LMBot
from helperbot.lr_scheduler import TriangularLR

MODEL_PATH = Path("data/cache/lm_unigram_dk_noseg/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 64
BPTT = 75


def get_tokens():
    tokens = joblib.load("data/tokens_unigram_False.pkl")
    # Filter out empty texts
    tokens = [x for x in tokens if x.shape[0] > 0]
    print_voc_stats(tokens)
    # Set shuffle = False to keep sentences from the same paragraph together
    trn_tokens, val_tokens = train_test_split(
        tokens, test_size=0.2, shuffle=False, random_state=898)
    val_tokens, tst_tokens = train_test_split(
        val_tokens, test_size=0.5, shuffle=False, random_state=898)
    voc_sz = int(np.max([np.max(x) for x in tokens]) + 1)
    return trn_tokens, val_tokens, tst_tokens, voc_sz


def print_voc_stats(tokens):
    total_tokens = np.sum([x.shape[0] for x in tokens])
    unks = np.sum([np.sum(x == 0) for x in tokens])
    print("Total tokens: %d\nUnknown Percentage: %.2f %%" %
          (total_tokens, unks * 100 / total_tokens))


def main():
    trn_tokens, val_tokens, tst_tokens, voc_sz = get_tokens()
    trn_loader = LanguageModelLoader(
        np.concatenate(trn_tokens), BATCH_SIZE, BPTT)
    val_loader = LanguageModelLoader(
        np.concatenate(val_tokens), BATCH_SIZE, BPTT, randomize=False)
    tst_loader = LanguageModelLoader(
        np.concatenate(tst_tokens), BATCH_SIZE, BPTT, randomize=False)
    model = get_language_model(
        voc_sz,
        emb_sz=500,
        pad_idx=2,
        dropoute=0,
        rnn_hid=500,
        rnn_layers=3,
        bidir=False,
        dropouth=0.1,
        dropouti=0.1,
        wdrop=0.05,
        qrnn=False,
        tie_weights=True
    )
    model = model.to("cuda:0")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-3, betas=(0.8, 0.999))
    bot = LMBot(
        model, trn_loader, val_loader,
        optimizer=optimizer,
        clip_grad=25.,
        log_dir=MODEL_PATH / "logs",
        checkpoint_dir=MODEL_PATH,
        echo=True,
        use_tensorboard=False,
        avg_window=len(trn_loader) // 10 * 2
    )
    n_steps = len(trn_loader) * 20
    bot.train(
        n_steps,
        log_interval=len(trn_loader) // 10,
        # eval after finished one epoch to maintain training contexts
        # TODO: cache training context before eval and resotre after eval
        snapshot_interval=len(trn_loader),
        min_improv=1e-3,
        scheduler=TriangularLR(
            optimizer, max_mul=64, ratio=5,
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
