"""M6 neural play model (PyTorch, CPU): trunk MLP + DeepSets player pooling + bin and turnover heads.

LICENSE: CC BY-SA 4.0. Trained on nflverse participation data (NFL Next Gen
Stats via nflverse 2016-2022, FTN Data via nflverse 2023+); the saved weights
and every prediction are CC BY-SA 4.0. Spec: context/ml-m6-method.md, section 5.

Architecture.
- Trunk input: the allowlisted features of the view (`data.FEATURES`),
  median-imputed with a missing flag for the columns that can be missing, and
  standardized with the training means and SDs.
- Players (variants N1, N2): every (side, player) seen in training gets an
  embedding (offense and defense are separate tables, like the M5b
  "off|id" / "def|id" keys), starting at zero; unseen players keep a fixed
  zero vector, so a player the data say little about stays "average". Each of
  the 11 offensive players goes through the same small MLP phi, and the 11
  outputs are averaged (DeepSets), so the order players are listed in cannot
  matter; the same for the 11 defenders with their own phi. The two pooled
  vectors join the trunk input.
- N2 also feeds phi each player's prior values as of the start of the play's
  season: his M5 box value, his M5b RAPM value, and a has-box-record flag
  (season-ahead: from data before that season only). The embedding is then
  a learned residual on top of the prior-driven part, pulled toward zero by
  the embedding penalty, i.e. regularized toward the M5/M5b values.
- Heads: 53 bin logits and one turnover logit, their biases started at the
  training marginals. The bin softmax is folded onto
  the bins possible from the play's yard line inside the loss (mass beyond
  the goal line goes to the touchdown bin), so training and scoring see the
  same distribution.
Loss: folded bin negative log likelihood + turnover binary cross-entropy +
`emb_l2` x the mean squared norm of the batch's player embeddings; AdamW with
weight decay on the trunk. Early stopping on the last training season picks
the number of epochs; the final model is refit on every training season for
that many epochs (same seed). Seeds are fixed and torch runs with
deterministic algorithms on a fixed number of CPU threads.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..players import participation as pt
from . import data as pdata

LICENSE_NOTE = pdata.LICENSE_NOTE
VARIANTS = ("N0", "N1", "N2")
MISSING_FLAGS = ("temp", "wind")  # missing only indoors (roof-driven); every other feature is filled in data.py
PRIOR_COLS = ("box", "rapm", "has_box")
THREADS = 4


def _torch():
    import torch
    return torch


def set_determinism(seed: int) -> None:
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


@dataclass(frozen=True)
class NetConfig:
    variant: str = "N1"
    hidden: int = 128
    emb_dim: int = 8
    phi: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    emb_l2: float = 1e-3
    batch: int = 512
    max_epochs: int = 40
    patience: int = 4
    seed: int = 20261003

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def players(self) -> bool:
        return self.variant in ("N1", "N2")

    @property
    def priors(self) -> bool:
        return self.variant == "N2"


# --------------------------------------------------------------------------- preprocessing

@dataclass
class Prep:
    feats: list
    med: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    flags: list

    @classmethod
    def fit(cls, table: pd.DataFrame, feats: list) -> "Prep":
        pdata.check_features(feats)
        X = table[feats].to_numpy(float)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        flags = [f for f in MISSING_FLAGS if f in feats]
        Z = cls._raw(X, med, feats, flags, table)
        mean = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        return cls(list(feats), med, mean, sd, flags)

    @staticmethod
    def _raw(X, med, feats, flags, table) -> np.ndarray:
        X = np.where(np.isnan(X), med[None, :], X)
        miss = [table[f].isna().to_numpy(float) for f in flags]
        return np.column_stack([X] + miss) if miss else X

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        X = table[self.feats].to_numpy(float)
        return ((self._raw(X, self.med, self.feats, self.flags, table) - self.mean) / self.sd).astype(np.float32)

    @property
    def width(self) -> int:
        return len(self.feats) + len(self.flags)


@dataclass
class Vocab:
    off: dict
    de: dict

    @classmethod
    def fit(cls, table: pd.DataFrame) -> "Vocab":
        o = np.unique(table[pt.OFF_COLS].to_numpy(object).astype(str))
        d = np.unique(table[pt.DEF_COLS].to_numpy(object).astype(str))
        return cls({k: i + 1 for i, k in enumerate(o)}, {k: i + 1 for i, k in enumerate(d)})

    def index(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        o = table[pt.OFF_COLS].to_numpy(object).astype(str)
        d = table[pt.DEF_COLS].to_numpy(object).astype(str)
        oi = pd.Series(self.off).reindex(o.ravel()).fillna(0).to_numpy(np.int64).reshape(o.shape)
        di = pd.Series(self.de).reindex(d.ravel()).fillna(0).to_numpy(np.int64).reshape(d.shape)
        return oi, di


def prior_arrays(table: pd.DataFrame, priors: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n, 11, 3) offense and defense prior features per play, each from its season's start-of-season table.

    `priors[season]` is indexed by "off|<id>" / "def|<id>" with PRIOR_COLS; missing players get zeros.
    """
    n = len(table)
    po = np.zeros((n, 11, len(PRIOR_COLS)), np.float32)
    pdf = np.zeros_like(po)
    seasons = table["season"].to_numpy(int)
    for S in np.unique(seasons):
        m = seasons == S
        lut = priors[int(S)][list(PRIOR_COLS)]
        for side, cols, out in (("off", pt.OFF_COLS, po), ("def", pt.DEF_COLS, pdf)):
            ids = np.char.add(f"{side}|", table.loc[m, cols].to_numpy(object).astype(str))
            v = lut.reindex(ids.ravel()).fillna(0.0).to_numpy(np.float32).reshape(ids.shape + (len(PRIOR_COLS),))
            out[m] = v
    return po, pdf


# --------------------------------------------------------------------------- the network

def build_net(cfg: NetConfig, n_x: int, n_off: int, n_def: int):
    torch = _torch()
    nn = torch.nn
    k = len(PRIOR_COLS) if cfg.priors else 0

    class PlayNet(nn.Module):
        def __init__(self):
            super().__init__()
            width = n_x
            if cfg.players:
                self.emb_off = nn.Embedding(n_off + 1, cfg.emb_dim, padding_idx=0)
                self.emb_def = nn.Embedding(n_def + 1, cfg.emb_dim, padding_idx=0)
                for e in (self.emb_off, self.emb_def):   # every player starts as "average" (zero vector)
                    nn.init.zeros_(e.weight)
                self.phi_off = nn.Sequential(nn.Linear(cfg.emb_dim + k, cfg.phi), nn.ReLU(), nn.Linear(cfg.phi, cfg.phi))
                self.phi_def = nn.Sequential(nn.Linear(cfg.emb_dim + k, cfg.phi), nn.ReLU(), nn.Linear(cfg.phi, cfg.phi))
                width += 2 * cfg.phi
            self.trunk = nn.Sequential(nn.Linear(width, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                       nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU())
            self.head_bins = nn.Linear(cfg.hidden, pdata.N_BINS)
            self.head_tov = nn.Linear(cfg.hidden, 1)

        def pooled(self, oi, di, po=None, pd_=None):
            eo, ed = self.emb_off(oi), self.emb_def(di)
            if po is not None:
                eo = torch.cat([eo, po], dim=-1)
                ed = torch.cat([ed, pd_], dim=-1)
            return self.phi_off(eo).mean(dim=1), self.phi_def(ed).mean(dim=1)

        def forward(self, x, oi=None, di=None, po=None, pd_=None):
            if cfg.players:
                a, b = self.pooled(oi, di, po, pd_)
                x = torch.cat([x, a, b], dim=1)
            h = self.trunk(x)
            return self.head_bins(h), self.head_tov(h).squeeze(1)

        def emb_penalty(self, oi, di):
            if not cfg.players:
                return torch.zeros(())
            return (self.emb_off(oi).pow(2).sum(dim=(1, 2)) + self.emb_def(di).pow(2).sum(dim=(1, 2))).mean()

    return PlayNet()


def init_heads(net, train: pd.DataFrame) -> None:
    """Start the output biases at the training marginals (log bin frequencies, logit of the turnover rate)."""
    torch = _torch()
    freq = np.bincount(train["bin"].to_numpy(int), minlength=pdata.N_BINS) + 1.0
    rate = (train["turnover"].sum() + 1.0) / (len(train) + 2.0)
    with torch.no_grad():
        net.head_bins.bias.copy_(torch.from_numpy(np.log(freq / freq.sum()).astype(np.float32)))
        net.head_tov.bias.fill_(float(np.log(rate / (1 - rate))))


def folded_nll(logits, fold_idx, target):
    """-log P(observed bin) after folding the softmax onto the play's possible bins (stable logsumexp form)."""
    torch = _torch()
    hit = fold_idx == target[:, None]
    num = torch.logsumexp(logits.masked_fill(~hit, float("-inf")), dim=1)
    return torch.logsumexp(logits, dim=1) - num


# --------------------------------------------------------------------------- fit / predict

@dataclass
class PlayModel:
    cfg: NetConfig
    view: str
    prep: Prep
    vocab: Vocab
    prior_scale: np.ndarray | None
    state: dict
    epochs: int
    history: list = field(default_factory=list)
    train_seconds: float = 0.0
    n_train: int = 0

    def _net(self):
        net = build_net(self.cfg, self.prep.width, len(self.vocab.off), len(self.vocab.de))
        net.load_state_dict(self.state)
        net.eval()
        return net

    def predict(self, table: pd.DataFrame, priors: dict | None = None, batch: int = 8192
                ) -> tuple[np.ndarray, np.ndarray]:
        """(n, 53) folded bin probabilities and P(turnover)."""
        torch = _torch()
        set_determinism(self.cfg.seed)
        net = self._net()
        t = _tensors(table, self.prep, self.vocab, self.cfg, priors, self.prior_scale)
        P, T = [], []
        with torch.no_grad():
            for s in range(0, len(table), batch):
                sl = slice(s, s + batch)
                logits, tl = net(*_inputs(t, sl, self.cfg))
                p = torch.softmax(logits, dim=1)
                f = torch.zeros_like(p).scatter_add_(1, t["fold"][sl], p)
                P.append(f.numpy())
                T.append(torch.sigmoid(tl).numpy())
        p = np.concatenate(P) if P else np.zeros((0, pdata.N_BINS))
        return pdata.fold(p, table["yardline_100"].to_numpy(float)), np.concatenate(T) if T else np.zeros(0)

    def save(self, path: Path, meta: dict | None = None) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"license": LICENSE_NOTE, "cfg": self.cfg.to_dict(), "view": self.view, "state": self.state,
                    "prep": asdict(self.prep), "vocab_off": self.vocab.off, "vocab_def": self.vocab.de,
                    "prior_scale": self.prior_scale, "epochs": self.epochs, "history": self.history,
                    "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: Path) -> "PlayModel":
        torch = _torch()
        d = torch.load(path, weights_only=False)
        prep = Prep(**d["prep"])
        return cls(NetConfig(**d["cfg"]), d["view"], prep, Vocab(d["vocab_off"], d["vocab_def"]), d["prior_scale"],
                   d["state"], d["epochs"], d["history"])

    def embedding_norms(self) -> pd.DataFrame:
        if not self.cfg.players:
            return pd.DataFrame(columns=["key", "norm"])
        rows = []
        for side, voc, name in (("off", self.vocab.off, "emb_off.weight"), ("def", self.vocab.de, "emb_def.weight")):
            w = self.state[name].numpy()
            keys = sorted(voc, key=voc.get)
            rows.append(pd.DataFrame({"key": [f"{side}|{k}" for k in keys],
                                      "norm": np.linalg.norm(w[[voc[k] for k in keys]], axis=1)}))
        return pd.concat(rows, ignore_index=True)


def _tensors(table, prep, vocab, cfg, priors, prior_scale) -> dict:
    torch = _torch()
    out = {"x": torch.from_numpy(prep.transform(table)),
           "fold": torch.from_numpy(pdata.fold_index(table["yardline_100"].to_numpy(float)).astype(np.int64))}
    if "bin" in table.columns:
        out["y"] = torch.from_numpy(table["bin"].to_numpy(np.int64, copy=True))
        out["tov"] = torch.from_numpy(table["turnover"].to_numpy(np.float32, copy=True))
    if cfg.players:
        oi, di = vocab.index(table)
        out["oi"], out["di"] = torch.from_numpy(oi.copy()), torch.from_numpy(di.copy())
    if cfg.priors:
        po, pd_ = prior_arrays(table, priors)
        out["po"] = torch.from_numpy(po / prior_scale[None, None, :].astype(np.float32))
        out["pd"] = torch.from_numpy(pd_ / prior_scale[None, None, :].astype(np.float32))
    return out


def _inputs(t: dict, sl, cfg: NetConfig) -> tuple:
    args = [t["x"][sl]]
    if cfg.players:
        args += [t["oi"][sl], t["di"][sl]]
        if cfg.priors:
            args += [t["po"][sl], t["pd"][sl]]
    return tuple(args)


def _prior_scale(table: pd.DataFrame, priors: dict) -> np.ndarray:
    po, pd_ = prior_arrays(table, priors)
    v = np.concatenate([po.reshape(-1, len(PRIOR_COLS)), pd_.reshape(-1, len(PRIOR_COLS))])
    sd = v.std(axis=0)
    return np.where(sd > 1e-9, sd, 1.0)


def _train(net, t: dict, cfg: NetConfig, epochs: int, val: dict | None = None):
    """Mini-batch training; with `val`, returns per-epoch validation losses and the best state."""
    torch = _torch()
    decay, no_decay, emb = [], [], []
    for name, prm in net.named_parameters():
        if name.startswith("emb_"):
            emb.append(prm)
        elif prm.ndim > 1:
            decay.append(prm)
        else:
            no_decay.append(prm)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if emb:
        groups.append({"params": emb, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    n = len(t["y"])
    gen = torch.Generator().manual_seed(cfg.seed)
    bce = torch.nn.BCEWithLogitsLoss()
    hist, best, best_state, bad = [], np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch]
            logits, tl = net(*_inputs(t, idx, cfg))
            loss = folded_nll(logits, t["fold"][idx], t["y"][idx]).mean() + bce(tl, t["tov"][idx])
            if cfg.players and cfg.emb_l2 > 0:
                loss = loss + cfg.emb_l2 * net.emb_penalty(t["oi"][idx], t["di"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        rec = {"epoch": ep + 1, "train_loss": tot / n}
        if val is not None:
            net.eval()
            with torch.no_grad():
                vl = []
                for s in range(0, len(val["y"]), 8192):
                    sl = slice(s, s + 8192)
                    lg, tl = net(*_inputs(val, sl, cfg))
                    vl.append(folded_nll(lg, val["fold"][sl], val["y"][sl]).sum().item()
                              + torch.nn.functional.binary_cross_entropy_with_logits(
                                  tl, val["tov"][sl], reduction="sum").item())
            rec["val_loss"] = sum(vl) / len(val["y"])
            if rec["val_loss"] < best - 1e-6:
                best, bad = rec["val_loss"], 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        if val is not None and bad >= cfg.patience:
            break
    return hist, best_state


def fit(train: pd.DataFrame, view: str, cfg: NetConfig, priors: dict | None = None,
        epochs: int | None = None, val: pd.DataFrame | None = None) -> PlayModel:
    """Train on `train`. With `val`, early-stop on it (up to cfg.max_epochs); else train exactly `epochs`."""
    t0 = time.perf_counter()
    feats = pdata.check_features(pdata.FEATURES[view])
    set_determinism(cfg.seed)
    prep = Prep.fit(train, feats)
    vocab = Vocab.fit(train) if cfg.players else Vocab({}, {})
    scale = _prior_scale(train, priors) if cfg.priors else None
    tt = _tensors(train, prep, vocab, cfg, priors, scale)
    tv = _tensors(val, prep, vocab, cfg, priors, scale) if val is not None else None
    net = build_net(cfg, prep.width, len(vocab.off), len(vocab.de))
    init_heads(net, train)
    hist, best_state = _train(net, tt, cfg, epochs if epochs is not None else cfg.max_epochs, tv)
    if val is not None:
        best_ep = int(min(hist, key=lambda r: r["val_loss"])["epoch"])
        state = best_state
    else:
        best_ep = int(epochs)
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return PlayModel(cfg, view, prep, vocab, scale, state, best_ep, hist, time.perf_counter() - t0, len(train))


TRUNK_GRID = ({"lr": 1e-3, "hidden": 128}, {"lr": 3e-4, "hidden": 128}, {"lr": 1e-3, "hidden": 256})
EMB_GRID = (1e-3, 1e-2, 1e-1)


def best_val(model: PlayModel) -> float:
    return float(min(r["val_loss"] for r in model.history))


def fit_season_ahead(train: pd.DataFrame, view: str, cfgs: list, priors: dict | None = None
                     ) -> tuple[PlayModel, dict]:
    """Each config is early-stopped on the last training season (fit on the earlier ones); the one with the
    lowest validation loss is refit on every training season for its early-stopped number of epochs."""
    t0 = time.perf_counter()
    last = int(train["season"].max())
    inner_tr, val = train[train["season"] < last], train[train["season"] == last]
    inner = [fit(inner_tr, view, c, priors, val=val) for c in cfgs]
    losses = [best_val(m) for m in inner]
    k = int(np.argmin(losses))
    final = fit(train, view, cfgs[k], priors, epochs=inner[k].epochs)
    info = {"grid": [c.to_dict() for c in cfgs], "val_loss": losses, "chosen": cfgs[k].to_dict(),
            "epochs": inner[k].epochs, "inner_history": inner[k].history, "final_seconds": final.train_seconds,
            "total_seconds": time.perf_counter() - t0, "n_train": int(len(train))}
    return final, info
