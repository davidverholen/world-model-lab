"""Rung-4 pre-check: does a frozen sentence encoder (MiniLM-L6) separate r1 recipe rule-variants?

Reads a dumped manual dataset (/tmp/rtfm_recipes.json — manuals + their gestures, correct &
swapped, train/eval content-disjoint splits, produced from crafter_rtfm) and asks: can a LINEAR
probe decode the recipe gesture (the ordered actions) from the pooled MiniLM embedding, on
HELD-OUT manuals? And do SWAPPED manuals decode to the SWAPPED gesture (the embedding tracks the
*displayed* rule, so conditioning on it would follow a swap)? Green → MiniLM suffices for
FiLM/concat conditioning; red (esp. later gesture positions blurring) → go token-level. Mirrors
the 0029/0031 diagnose-before-build method. See knowledge/design/rung4-manual-conditioned-agent.md.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer


def _onehot(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y), len(classes)))
    for i, v in enumerate(y):
        Y[i, idx[v]] = 1.0
    return Y, idx


def multiclass_probe(Xtr, ytr, Xte, classes, lam=10.0):
    """Linear (ridge→one-hot, argmax) multi-class probe; returns predicted class per test row."""
    Y, idx = _onehot(ytr, classes)
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte1 = np.hstack([Xte, np.ones((len(Xte), 1))])
    W = np.linalg.solve(Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1]), Xtr1.T @ Y)
    inv = {i: c for c, i in idx.items()}
    return np.array([inv[i] for i in (Xte1 @ W).argmax(1)])


def main():
    with open("/tmp/rtfm_recipes.json") as f:
        rows = json.load(f)
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    Ec = enc.encode([r["correct"] for r in rows], convert_to_numpy=True, normalize_embeddings=True)
    Es = enc.encode([r["swapped"] for r in rows], convert_to_numpy=True, normalize_embeddings=True)
    split = np.array([r["split"] for r in rows])
    tr, ev = split == "train", split == "eval"
    Ec, Es = Ec.astype(np.float64), Es.astype(np.float64)

    print(f"MiniLM-L6 (384-d) | {tr.sum()} train / {ev.sum()} held-out recipe manuals\n")
    print("decode recipe gesture[k] from the manual embedding (held-out eval):")
    for k in range(3):
        yk = np.array([r["gesture_correct"][k] for r in rows])
        classes = sorted(set(yk))
        pred = multiclass_probe(Ec[tr], yk[tr], Ec[ev], classes)
        acc = (pred == yk[ev]).mean()
        chance = max(np.mean(yk[ev] == c) for c in classes)
        print(f"  gesture[{k}]  acc={acc:.2f}  (chance {chance:.2f}, {len(classes)} classes)")

    # swap-tracking: probe trained on CORRECT manuals; on SWAPPED manuals does it predict the
    # SWAPPED gesture (tracks the displayed rule) rather than the correct one?
    print("\nswap-tracking (probe trained on correct; applied to SWAPPED held-out manuals):")
    for k in range(3):
        yk_c = np.array([r["gesture_correct"][k] for r in rows])
        yk_s = np.array([r["gesture_swapped"][k] for r in rows])
        classes = sorted(set(yk_c) | set(yk_s))
        pred = multiclass_probe(Ec[tr], yk_c[tr], Es[ev], classes)
        follows_swap = (pred == yk_s[ev]).mean()
        matches_correct = (pred == yk_c[ev]).mean()
        print(f"  gesture[{k}]  →swapped={follows_swap:.2f}  →correct={matches_correct:.2f} "
              f"(want high swapped, low correct)")

    # token-level confirmation: is the recipe info in the TOKENS (pooling kills it)? mean+max over
    # token embeddings is a crude token-preserving readout — lift ⇒ go token-level.
    def tokpool(texts):
        toks = enc.encode(texts, output_value="token_embeddings")  # list of (n_tok, d) tensors
        out = []
        for t in toks:
            a = t.cpu().numpy()
            out.append(np.concatenate([a.mean(0), a.max(0)]))
        return np.asarray(out, dtype=np.float64)

    Tc = tokpool([r["correct"] for r in rows])
    print("\n[token-level mean+max] decode gesture[k] (held-out eval) — token-level recovery?")
    for k in range(3):
        yk = np.array([r["gesture_correct"][k] for r in rows])
        classes = sorted(set(yk))
        pred = multiclass_probe(Tc[tr], yk[tr], Tc[ev], classes)
        print(f"  gesture[{k}]  acc={(pred == yk[ev]).mean():.2f}")


if __name__ == "__main__":
    main()
