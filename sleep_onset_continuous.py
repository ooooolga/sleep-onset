import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
import mne
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sleep_onset_def_v9 as v9
import sleep_onset_instab_v11 as v11
import sleep_onset_controls_v12 as v12
import sleep_onset_pzoz_v15 as v15
import sleep_onset_scale_v16 as v16

CACHE_FILE   = "features_v19.pkl"
N_RECORDINGS = 80
TAG_CH       = "EEG Pz-Oz"          # settled in v16/v17
WIN_SEC, STEP_SEC = v11.WIN_SEC, v11.STEP_SEC

# leg -> (regression window, calibration window), minutes relative to onset
LEGS = {
    "REAL":    ((-25.0,   0.0), (-40.0, -30.0)),
    "PLACEBO": ((-55.0, -30.0), (-70.0, -60.0)),
}
MIN_REG_WINDOWS  = 60
MIN_BASE_WINDOWS = 30
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
N_INNER = 5


# ----------------------------------------------------------------------
# Extraction: 4 s spectra tagged with minutes-to-onset
# ----------------------------------------------------------------------
def process_recording(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if TAG_CH not in raw.ch_names:
        return None
    raw.pick([TAG_CH])
    sf = raw.info["sfreq"]
    sig = mne.filter.filter_data(raw.get_data()[0] * 1e6, sf, 0.3, 45.0,
                                 verbose=False)

    annot = mne.read_annotations(hyp_path)
    total = int(len(sig) / sf)
    hypno = np.full(total, np.nan)
    for a in annot:
        st = v9.STAGE_MAP.get(a["description"], np.nan)
        s = int(a["onset"]); e = int(a["onset"] + a["duration"])
        hypno[s:min(e, total)] = st
    is_sleep = (hypno < 0) & (~np.isnan(hypno))
    if not is_sleep.any():
        return None
    onset = v9.find_onset(is_sleep, v11.ONSET_PERSIST_MIN, v11.ONSET_PERSIST_FRAC)
    if onset is None:
        return None

    win, step = int(WIN_SEC * sf), int(STEP_SEC * sf)
    store = {k: {"X": [], "t": [], "B": [], "n": 0, "rej": 0} for k in LEGS}
    for start in range(0, len(sig) - win, step):
        centre  = (start + win / 2) / sf
        rel_min = (centre - onset) / 60.0
        roles = []
        for leg, ((r0, r1), (b0, b1)) in LEGS.items():
            if r0 <= rel_min < r1:
                roles.append((leg, "reg"))
            elif b0 <= rel_min < b1:
                roles.append((leg, "base"))
        if not roles:
            continue
        c = int(centre)
        if c >= len(hypno) or hypno[c] != 0:      # scored wake only
            continue
        w = sig[start:start + win]
        for leg, kind in roles:
            if kind == "reg":
                store[leg]["n"] += 1
        if v9.is_artifact(w):
            for leg, kind in roles:
                if kind == "reg":
                    store[leg]["rej"] += 1
            continue
        spec = v9.spectrum(w, sf)
        for leg, kind in roles:
            if kind == "reg":
                store[leg]["X"].append(spec); store[leg]["t"].append(rel_min)
            else:
                store[leg]["B"].append(spec)

    nb = len(v9.freq_axis(sf))
    out = {}
    for leg, d in store.items():
        out[leg] = {
            "X": np.array(d["X"]) if d["X"] else np.zeros((0, nb)),
            "t": np.array(d["t"]) if d["t"] else np.zeros(0),
            "B": np.array(d["B"]) if d["B"] else np.zeros((0, nb)),
            "rej_pct": 100.0 * d["rej"] / max(d["n"], 1),
        }
    out["onset_min"] = onset / 60.0
    return out


def extract_all(pairs):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            cache = {r["name"]: r for r in pickle.load(fh)}
        print(f"Loaded {len(cache)} cached recordings from {CACHE_FILE}")
    recs, new = [], 0
    for psg, hyp in pairs:
        if psg in cache:
            recs.append(cache[psg]); continue
        try:
            r = process_recording(v9.download(psg), v9.download(hyp))
        except Exception as ex:
            print(f"  {psg} ERROR {ex}"); continue
        if r is None:
            continue
        r["name"] = psg; r["subject"] = v9.subject_id(psg)
        recs.append(r); new += 1
        if new % 10 == 0:
            print(f"  extracted {new} new")
    if new:
        with open(CACHE_FILE, "wb") as fh:
            pickle.dump(recs, fh)
        print(f"Cached {len(recs)} recordings to {CACHE_FILE}")
    return recs


# ----------------------------------------------------------------------
# Design matrices
# ----------------------------------------------------------------------
def build(rec, leg):
    d = rec[leg]
    if len(d["B"]) < MIN_BASE_WINDOWS or len(d["X"]) < MIN_REG_WINDOWS:
        return None
    mu, sd = d["B"].mean(0), d["B"].std(0) + 1e-9
    X = (d["X"] - mu) / sd
    if not np.isfinite(X).all():
        return None
    return X, d["t"]


def fit_ridge(train, alpha):
    X = np.vstack([r["_X"] for r in train])
    t = np.concatenate([r["_t"] for r in train])
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=alpha).fit(sc.transform(X), t)
    return sc, m


def choose_alpha(train, rng):
    """Grouped inner CV on the training subjects only."""
    subs = sorted(set(r["subject"] for r in train))
    rng.shuffle(subs)
    folds = [subs[i::N_INNER] for i in range(N_INNER)]
    best_a, best = ALPHAS[0], -np.inf
    for a in ALPHAS:
        scores = []
        for f in folds:
            itr = [r for r in train if r["subject"] not in f]
            ite = [r for r in train if r["subject"] in f]
            if not ite or len(set(r["subject"] for r in itr)) < 2:
                continue
            sc, m = fit_ridge(itr, a)
            for r in ite:
                pred = m.predict(sc.transform(r["_X"]))
                if np.std(pred) > 1e-9:
                    scores.append(spearmanr(pred, r["_t"])[0])
        if scores and np.mean(scores) > best:
            best, best_a = np.mean(scores), a
    return best_a


def evaluate(usable, seed=0):
    """Leave-one-subject-out; Spearman computed WITHIN each recording."""
    rng = np.random.default_rng(seed)
    per_rec = {}
    for held in sorted(set(r["subject"] for r in usable)):
        tr = [r for r in usable if r["subject"] != held]
        te = [r for r in usable if r["subject"] == held]
        a = choose_alpha(tr, rng)
        sc, m = fit_ridge(tr, a)
        for r in te:
            pred = m.predict(sc.transform(r["_X"]))
            per_rec[r["name"]] = (spearmanr(pred, r["_t"])[0]
                                  if np.std(pred) > 1e-9 else 0.0)
    return per_rec


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    pairs = v9.pair_psg_hypno(v9.local_file_list())[:N_RECORDINGS]
    recs = extract_all(pairs)
    print(f"\n{len(recs)} recordings, "
          f"{len(set(r['subject'] for r in recs))} subjects\n")

    usable = {}
    for leg in LEGS:
        keep = []
        for r in recs:
            if r[leg]["rej_pct"] > v9.REJECT_MAX_PCT:
                continue
            if build(r, leg) is None:
                continue
            keep.append(r["name"])
        usable[leg] = set(keep)
        print(f"  {leg:<8s} usable: {len(keep)}/{len(recs)} recordings")
    common = usable["REAL"] & usable["PLACEBO"]
    kept = [r for r in recs if r["name"] in common]
    subs = sorted(set(r["subject"] for r in kept))
    print(f"  usable in BOTH: {len(kept)} recordings / {len(subs)} subjects\n")
    if len(subs) < 8:
        print("Too few subjects; stopping.")
        return

    per_leg = {}
    for leg in LEGS:
        u = []
        for r in kept:
            X, t = build(r, leg)
            s = dict(r); s["_X"], s["_t"] = X, t
            u.append(s)
        per_leg[leg] = evaluate(u)

    # per-subject aggregation (mean over that subject's recordings)
    by_sub = {}
    for r in kept:
        n = r["name"]
        if n in per_leg["REAL"] and n in per_leg["PLACEBO"]:
            by_sub.setdefault(r["subject"], []).append(
                (per_leg["REAL"][n], per_leg["PLACEBO"][n]))
    ss = sorted(by_sub)
    real = np.array([np.mean([a for a, _ in by_sub[s]]) for s in ss])
    plac = np.array([np.mean([b for _, b in by_sub[s]]) for s in ss])
    cons = real - np.maximum(plac, 0.0)

    print("=" * 78)
    print("CONTINUOUS TARGET: predicting minutes-to-onset by ridge regression")
    print("=" * 78)
    print(f"  within-recording Spearman rho, averaged over {len(ss)} subjects")
    print(f"    REAL     (-25..0 min)    {real.mean():+.3f}  (SEM {real.std()/np.sqrt(len(real)):.3f})")
    print(f"    PLACEBO  (-55..-30 min)  {plac.mean():+.3f}  (SEM {plac.std()/np.sqrt(len(plac)):.3f})")
    print(f"    conservative net         {cons.mean():+.3f}  "
          f"(SEM {cons.std()/np.sqrt(len(cons)):.3f}), "
          f"positive in {int((cons > 0).sum())}/{len(cons)}, "
          f"p = {wilcoxon(cons)[1]:.2e}")
    print("\n  PLACEBO regresses on minutes-to-onset over a stretch where the")
    print("  person's state is NOT changing, so it measures what the passage of")
    print("  time alone buys.  Only the difference is evidence of drowsiness.")

    # ---------- is the placebo doing a lot of work here? ----------
    print(f"\n  Time alone accounts for "
          f"{100.0 * max(plac.mean(), 0) / max(real.mean(), 1e-9):.0f}% "
          f"of the raw REAL correlation.")

    # ---------- comparison with the binary formulation ----------
    print("\n" + "=" * 78)
    print("Is the continuous target worth the extra machinery?")
    print("=" * 78)
    print("  Binary  (v17, SPEC-full noPCA, Pz-Oz):  conservative NET  +0.164 AUC")
    print(f"  Continuous (this round):                conservative net  "
          f"{cons.mean():+.3f} Spearman rho")
    print("  These are different scales and cannot be subtracted.  What CAN be")
    print("  compared is how far each clears its own placebo relative to its own")
    print("  spread, so both are put on a common footing below (Cohen's d over")
    print("  subjects of the per-subject net):")
    d_cont = cons.mean() / (cons.std() + 1e-12)
    print(f"    continuous  d = {d_cont:.2f}  (n = {len(cons)})")
    d_bin = 0.164 / (0.026 * np.sqrt(41))   # v17 net and SEM, n = 41
    print(f"    binary      d = {d_bin:.2f}  (v17: +0.164 net, SEM 0.026, n = 41)")
    print("  Caveat: the continuous task regresses over a 25-minute span while")
    print("  the binary one uses two short bands, so it sees several times more")
    print("  windows per recording.  Part of the larger d is that statistical")
    print("  advantage, not a physiological one.")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar([0, 1], [real.mean(), plac.mean()], 0.55,
                color=["#2E8B57", "#C0392B"],
                yerr=[real.std() / np.sqrt(len(real)),
                      plac.std() / np.sqrt(len(plac))], capsize=5)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["REAL\n(-25..0 min)", "PLACEBO\n(-55..-30 min)"])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("within-recording Spearman rho")
    axes[0].set_title("Predicting minutes-to-onset\n(bars = SEM over subjects)")
    axes[0].grid(alpha=0.3, axis="y")

    o = np.argsort(-cons)
    axes[1].bar(range(len(cons)), cons[o],
                color=["#2E8B57" if q > 0 else "#C0392B" for q in cons[o]])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("subject (ranked)")
    axes[1].set_ylabel("conservative net rho")
    axes[1].set_title(f"Per subject: REAL minus PLACEBO\n"
                      f"positive in {int((cons > 0).sum())}/{len(cons)}")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("continuous_v19.png", dpi=130)
    print("\nSaved figure: continuous_v19.png")


if __name__ == "__main__":
    main()
