"""
InteraCare - training pipeline (reference implementation)

NOTE: this script was RECONSTRUCTED from the logic that ships inside the app
(js/core.js feature code + the "Model & pipeline" screen). It is not the
original notebook, so exact metrics can differ by a point or two. The 13
features are a line-by-line port of feats() in js/core.js and were checked
against the JavaScript on sample drug pairs.

Pipeline (same 8 steps as the app's "Workflow" card):
  1. dataset collection   - 8 DDInter CSVs (or data/ddinter_pairs.csv)
  2. preprocessing        - order-independent pair keys, drop duplicates
  3. feature extraction   - 13 interaction-network features per drug pair
  4. model training       - gradient boosting, 120 rounds, depth 5, class-balanced
  5. validation           - 80/20 split on graded pairs (Minor/Moderate/Major)
  6. prediction           - class probabilities
  7. risk classification  - label + 0-100 risk index (done in the browser)
  8. dashboard            - export model to js/data/model-data.js

Usage
  python train_model.py                                  # uses ../data/ddinter_pairs.csv
  python train_model.py --ddinter-dir path/to/ddinter    # raw DDInter downloads (code_A..code_V)
  python train_model.py --out ../js/data/model-data.js   # overwrite the model used by the website
"""
import argparse, glob, json, math, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

LEVEL = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Major": 3}
FEATURE_NAMES = [
    "Min drug connectivity", "Max drug connectivity", "Mean major share",
    "Max major share", "Mean moderate share", "Mean minor share",
    "Partner overlap (Jaccard)", "Major-partner overlap", "Common partners",
    "Avg severity to common partners", "Common partners both major",
    "Common partners with a major link", "Graded share of common partners",
]
N_ROUNDS, MAX_DEPTH, LEARNING_RATE, SEED = 120, 5, 0.12, 42


# ---------------------------------------------------------------- 1-2. data
def load_pairs(ddinter_dir, pairs_csv):
    if ddinter_dir:
        files = sorted(glob.glob(os.path.join(ddinter_dir, "ddinter_downloads_code_*.csv")))
        if not files:
            raise SystemExit(f"No ddinter_downloads_code_*.csv files in {ddinter_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"collected {len(df):,} rows from {len(files)} DDInter files")
    else:
        df = pd.read_csv(pairs_csv)
        print(f"loaded {len(df):,} pairs from {pairs_csv}")
    df = df[["Drug_A", "Drug_B", "Level"]].copy()
    df["Level"] = df["Level"].map(LEVEL)
    df = df.dropna()
    df = df[df.Drug_A != df.Drug_B]
    # order-independent key: (A,B) and (B,A) are the same pair
    lo = df[["Drug_A", "Drug_B"]].min(axis=1)
    hi = df[["Drug_A", "Drug_B"]].max(axis=1)
    df = pd.DataFrame({"a": lo, "b": hi, "level": df["Level"].astype(int)})
    before = len(df)
    df = df.drop_duplicates(["a", "b"]).reset_index(drop=True)
    print(f"removed {before - len(df):,} duplicate pairs -> {len(df):,} unique pairs")
    return df


# ------------------------------------------------------- 3. network features
class Graph:
    def __init__(self, df):
        self.names = sorted(set(df.a) | set(df.b))
        idx = {n: i for i, n in enumerate(self.names)}
        self.nb = [dict() for _ in self.names]          # drug -> {partner: level}
        for a, b, l in zip(df.a, df.b, df.level):
            i, j = idx[a], idx[b]
            self.nb[i][j] = l
            self.nb[j][i] = l
        # S[x] = [major, moderate, minor, unknown] counts (same order as the app)
        self.S = [[0, 0, 0, 0] for _ in self.names]
        for i, d in enumerate(self.nb):
            for l in d.values():
                self.S[i][3 - l] += 1
        self.idx = idx

    def feats(self, i, j):
        """Port of feats(i,j) from js/core.js. A pair's own record is removed
        from the counts so the model can't see the label it is predicting."""
        nb, S = self.nb, self.S
        own = nb[i].get(j)
        o = own is not None

        def st(x):
            c = S[x][:]
            if o:
                c[3 - own] -= 1
            tot = c[0] + c[1] + c[2] + c[3]
            g = c[0] + c[1] + c[2]
            return [tot, c[0], c[0] / g if g else 0, c[1] / g if g else 0, c[2] / g if g else 0]

        a, b = st(i), st(j)
        A, B = nb[i], nb[j]
        if len(A) > len(B):
            A, B = B, A
        cnt = ng = b3 = e3 = 0
        sl = 0
        for k, la in A.items():
            lb = B.get(k)
            if lb is None:
                continue
            cnt += 1
            if la > 0 and lb > 0:
                ng += 1
                sl += la + lb
            if la == 3 and lb == 3:
                b3 += 1
            if la == 3 or lb == 3:
                e3 += 1
        un = len(nb[i]) + len(nb[j]) - 2 * int(o) - cnt
        mu = a[1] + b[1] - b3
        return [
            math.log1p(min(a[0], b[0])), math.log1p(max(a[0], b[0])),
            (a[2] + b[2]) / 2, max(a[2], b[2]), (a[3] + b[3]) / 2, (a[4] + b[4]) / 2,
            cnt / un if un > 0 else 0, b3 / mu if mu > 0 else 0, math.log1p(cnt),
            sl / (2 * ng) / 3 if ng else 0, b3 / cnt if cnt else 0,
            e3 / cnt if cnt else 0, ng / cnt if cnt else 0,
        ]


# ---------------------------------------------------------------- 4-5. train
def metrics(model, X, y):
    p = model.predict(X)
    return {
        "acc": float(accuracy_score(y, p)),
        "f1": float(f1_score(y, p, average="macro")),
        "rec": [float(v) for v in recall_score(y, p, average=None)],
        "cm": confusion_matrix(y, p).tolist(),
    }


def export_tree(t):
    t = t.tree_
    left = [int(v) for v in t.children_left]            # -1 marks a leaf (same as the JS)
    return [
        [int(v) if left[n] != -1 else 0 for n, v in enumerate(t.feature)],
        [round(float(v), 6) if left[n] != -1 else 0 for n, v in enumerate(t.threshold)],
        left,
        [int(v) for v in t.children_right],
        [round(float(v), 6) for v in t.value[:, 0, 0]],
    ]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddinter-dir")
    ap.add_argument("--pairs", default=os.path.join(here, "..", "data", "ddinter_pairs.csv"))
    ap.add_argument("--out", default=os.path.join(here, "output", "model-data.js"))
    ap.add_argument("--skip-baselines", action="store_true", help="skip logistic regression / random forest")
    args = ap.parse_args()

    df = load_pairs(args.ddinter_dir, args.pairs)
    g = Graph(df)
    print(f"{len(g.names):,} drugs, graph built")

    graded = df[df.level > 0].reset_index(drop=True)      # Minor/Moderate/Major only
    t0 = time.time()
    X = np.array([g.feats(g.idx[a], g.idx[b]) for a, b in zip(graded.a, graded.b)])
    y = graded.level.to_numpy() - 1                       # 0=Minor 1=Moderate 2=Major
    print(f"features {X.shape} in {time.time() - t0:.0f}s")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    w = compute_sample_weight("balanced", ytr)            # class-balanced
    print(f"train {len(ytr):,} / test {len(yte):,}")

    res = {}
    if not args.skip_baselines:
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(Xtr, ytr)
        res["Logistic regression"] = metrics(lr, Xte, yte)
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, ytr)
        res["Random forest"] = metrics(rf, Xte, yte)

    gb = GradientBoostingClassifier(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                    learning_rate=LEARNING_RATE, random_state=SEED)
    gb.fit(Xtr, ytr, sample_weight=w)
    res["Gradient boosting"] = metrics(gb, Xte, yte)
    for k, v in res.items():
        print(f"  {k:20s} acc {v['acc']:.3f}  macro-F1 {v['f1']:.3f}  recall {[round(r, 2) for r in v['rec']]}")

    # ----------------------------------------------------------- 8. export
    MD = {
        "trees": [[export_tree(gb.estimators_[r][c]) for c in range(3)] for r in range(N_ROUNDS)],
        "init": [round(float(v), 8) for v in gb._raw_predict_init(Xte[:1])[0]],
        "lr": LEARNING_RATE,
        "fn": FEATURE_NAMES,
        "fi": [[n, round(float(v), 4)] for n, v in zip(FEATURE_NAMES, gb.feature_importances_)],
        "res": res,
        "ntrain": int(len(ytr)),
        "ntest": int(len(yte)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("/* Trained gradient-boosting model exported for the browser: init, lr, trees, feature importance (fi), comparison results (res), ntrain/ntest */\n")
        f.write("const MD=" + json.dumps(MD, separators=(",", ":")) + ";\n")
    print(f"model written to {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")
    np.save(os.path.join(os.path.dirname(os.path.abspath(args.out)), "_xte.npy"), Xte[:2000])


if __name__ == "__main__":
    main()
