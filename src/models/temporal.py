"""B05 — small torch temporal models (1D-CNN, GRU) on minute-bar sequences.

RTX 3050 sizing discipline (macro_gpu_lab TORCH_* ethos): seq_len 60, hidden
<= 64, 1-2 layers, dropout 0.3, weight decay, early stopping on a
chronological tail of train — everything modest, nothing close to 6 GB.
Sequences are built per symbol so a window never spans two symbols; windows
never cross into the test period (the harness's fit_predict contract gives
train and test rows separately).

CLI: .venv\\Scripts\\python.exe -m src.models.temporal [--register-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.experiments import registry  # noqa: E402
from src.gpu import get_device, set_seed  # noqa: E402
from src.models.evaluate import run_battery  # noqa: E402

BATTERY = "B05_temporal_core7"
FAMILY = "intraday_core7_1min"
SYMBOLS = ["SPY", "QQQ", "IWM", "BND"]

SEQ_LEN = 60
HIDDEN = 64
DROPOUT = 0.3
BATCH = 1024
MAX_EPOCHS = 20
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
ARCHS = ["cnn", "gru"]


def _make_sequences(X: np.ndarray, y: np.ndarray | None, seq_len: int):
    """Overlapping windows ending at each row (rows assumed chronological)."""
    from numpy.lib.stride_tricks import sliding_window_view

    if len(X) < seq_len:
        return None, None
    wins = sliding_window_view(X, (seq_len, X.shape[1])).squeeze(1)  # (n-s+1, s, f)
    ys = y[seq_len - 1:] if y is not None else None
    return wins, ys


def _build_model(arch: str, n_features: int):
    import torch.nn as nn

    if arch == "cnn":
        return nn.Sequential(
            nn.Conv1d(n_features, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Conv1d(HIDDEN, HIDDEN, kernel_size=5, padding=2, stride=2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(HIDDEN, 1),
        )

    class GRUHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, HIDDEN, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):  # x: (B, seq, feat)
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1]))

    return GRUHead()


def make_fit_predict(arch: str):
    def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn

        set_seed(config.SEED)
        device = get_device()
        # sanitize: temporal nets hate NaN; impute train medians
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med).astype(np.float32)
        Xte = np.where(np.isfinite(Xte), Xte, med).astype(np.float32)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

        seq_tr, y_seq = _make_sequences(Xtr, ytr, SEQ_LEN)
        if seq_tr is None:
            return np.full(len(Xte), 0.5)
        cut = int(len(seq_tr) * 0.85)  # chronological early-stop tail
        model = _build_model(arch, Xtr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.BCEWithLogitsLoss()

        def _epoch(lo, hi, train: bool) -> float:
            total, count = 0.0, 0
            model.train(train)
            for i in range(lo, hi, BATCH):
                xb = torch.from_numpy(seq_tr[i:i + BATCH]).to(device)
                yb = torch.from_numpy(y_seq[i:i + BATCH].astype(np.float32)).to(device)
                if arch == "cnn":
                    xb = xb.transpose(1, 2)  # (B, feat, seq)
                with torch.set_grad_enabled(train):
                    out = model(xb).squeeze(-1)
                    loss = lossf(out, yb)
                    if train:
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)
            return total / max(count, 1)

        best, best_state, bad = np.inf, None, 0
        for _ in range(MAX_EPOCHS):
            _epoch(0, cut, train=True)
            vl = _epoch(cut, len(seq_tr), train=False)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        # test windows: prepend the last (SEQ_LEN-1) train rows so every test
        # row gets a full backward-looking window (train rows are past data)
        joined = np.vstack([Xtr[-(SEQ_LEN - 1):], Xte])
        seq_te, _ = _make_sequences(joined, None, SEQ_LEN)
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(seq_te), BATCH):
                xb = torch.from_numpy(seq_te[i:i + BATCH]).to(device)
                if arch == "cnn":
                    xb = xb.transpose(1, 2)
                probs.append(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy())
        return np.concatenate(probs)

    return fit_predict


def ensure_registered() -> None:
    try:
        registry.load_registration(BATTERY)
    except registry.UnregisteredBattery:
        registry.register(
            BATTERY,
            hypothesis=(
                "Sequential structure in the last 60 minute-bars (order of "
                "moves, not just rolling summaries) carries triple-barrier "
                "signal the pointwise models (B01/B02) cannot see."
            ),
            dataset_family=FAMILY,
            horizons=config.HORIZONS_BARS,
            model="torch-temporal",
            param_grid={"arch": ARCHS, "seq_len": SEQ_LEN, "hidden": HIDDEN},
            n_trials=len(ARCHS) * len(config.HORIZONS_BARS),
        )


def main(register_only: bool = False) -> None:
    set_seed(config.SEED)
    ensure_registered()
    if register_only:
        print(f"{BATTERY} registered.")
        return
    grid = [({"model": f"temporal-{a}", "seq_len": SEQ_LEN, "hidden": HIDDEN},
             make_fit_predict(a)) for a in ARCHS]
    res = run_battery(BATTERY, grid, SYMBOLS)
    print(json.dumps({"status": res["status"],
                      "scorecard": res["scorecard_path"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    main(ap.parse_args().register_only)
