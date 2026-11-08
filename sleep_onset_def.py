import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import spearmanr, wilcoxon
import mne
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR   = "sleepedf_data"
CACHE_FILE = "features_v9.pkl"   # delete this file to force re-extraction
BASE_URL   = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
CHANNEL    = "EEG Fpz-Cz"
WIN_SEC    = 4.0
STEP_SEC   = 2.0
N_RECORDINGS    = 20            # 20 recordings ~ 10 real subjects
ARTIFACT_PTP_UV = 200.0
REJECT_MAX_PCT  = 15.0

# --- state-defining time windows (minutes relative to sleep onset) ---
TARGET_FROM,    TARGET_TO    =  -3.0,   0.0   # "about to fall asleep"
NONTARGET_FROM, NONTARGET_TO = -10.0,  -5.0   # "alert wake"
# the -5..-3 gap is a deliberate buffer, left unlabeled

# --- personal calibration segment (label-free, outside the evaluated bands) ---
BASELINE_FROM, BASELINE_TO = -25.0, -15.0
MIN_BASELINE_WINDOWS = 30

# --- THE THING WE ARE TESTING THIS ROUND ---
# name -> (persistence window in minutes, required sleep fraction inside it)
# frac = 0.0 reproduces the old "first sleep second" rule exactly.
ONSET_DEFS = {
    "OLD":       (10.0, 0.00),
    "PERSIST80": (10.0, 0.80),
    "LPS100":    (10.0, 1.00),
}

SPEC_FMIN, SPEC_FMAX = 0.5, 30.0
N_PCA = 2

STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": -1, "Sleep stage 2": -2,
    "Sleep stage 3": -3, "Sleep stage 4": -4, "Sleep stage R": -0.5,
}

# ----------------------------------------------------------------------
# File handling
# ----------------------------------------------------------------------
def subject_id(psg_name):
    """SC4ssNE0-PSG.edf -> 'ss'.  Both nights of one person share this id."""
    return psg_name[3:5]

def local_file_list():
    return sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []

def get_file_list():
    r = requests.get("https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt", timeout=30)
    r.raise_for_status()
    return [os.path.basename(p.split()[-1]) for p in r.text.splitlines()
            if len(p.split()) >= 2 and p.split()[-1].startswith("sleep-cassette/")]

def pair_psg_hypno(all_files):
    psgs   = sorted(f for f in all_files if f.endswith("-PSG.edf"))
    hypnos = [f for f in all_files if f.endswith("-Hypnogram.edf")]
    return [(psg, m[0]) for psg in psgs
            for m in [[h for h in hypnos if h.startswith(psg[:6])]] if m]

def download(fname):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        return path
    r = requests.get(BASE_URL + fname, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return path

# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def spectrum(data, sf):
    """Log power in ~0.5 Hz bins from SPEC_FMIN to SPEC_FMAX."""
    nperseg = min(len(data), int(sf * 2))
    freqs, psd = welch(data, sf, nperseg=nperseg)
    mask = (freqs >= SPEC_FMIN) & (freqs <= SPEC_FMAX)
    return np.log(psd[mask] + 1e-12)

def freq_axis(sf=100.0):
    nperseg = int(sf * 2)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sf)
    return freqs[(freqs >= SPEC_FMIN) & (freqs <= SPEC_FMAX)]

def is_artifact(w):
    return (w.max() - w.min()) > ARTIFACT_PTP_UV

# ----------------------------------------------------------------------
# Onset definitions
# ----------------------------------------------------------------------
def find_onset(is_sleep, persist_min, persist_frac):
    """First sleep second t such that the next `persist_min` minutes contain at
    least `persist_frac` sleep.  With persist_frac = 0 this is just the first
    sleep second (the old rule).  Returns None if no second qualifies.

    Uses a cumulative sum so the scan is O(n) rather than O(n * window)."""
    n = len(is_sleep)
    w = int(persist_min * 60)
    csum = np.concatenate([[0], np.cumsum(is_sleep.astype(np.int64))])
    for t in np.where(is_sleep)[0]:
        end = min(t + w, n)
        if end - t < w:
            continue          # too close to the end to verify persistence
        if (csum[end] - csum[t]) / float(w) >= persist_frac - 1e-9:
            return int(t)
    return None

def role_of(rel_min):
    """Which band does a window centre fall in, given minutes-to-onset?"""
    if TARGET_FROM    <= rel_min < TARGET_TO:    return "target"
    if NONTARGET_FROM <= rel_min < NONTARGET_TO: return "nontarget"
    if BASELINE_FROM  <= rel_min < BASELINE_TO:  return "baseline"
    return None

# ----------------------------------------------------------------------
# Per-recording processing: one pass, roles assigned under every definition
# ----------------------------------------------------------------------
def process_recording(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if CHANNEL not in raw.ch_names:
        return None
    raw.pick([CHANNEL])
    sf = raw.info["sfreq"]
    raw.filter(l_freq=0.3, h_freq=45.0, verbose=False)
    signal = raw.get_data()[0] * 1e6

    annot = mne.read_annotations(hyp_path)
    total_sec = int(len(signal) / sf)
    hypno = np.full(total_sec, np.nan)
    for a in annot:
        st = STAGE_MAP.get(a["description"], np.nan)
        s = int(a["onset"]); e = int(a["onset"] + a["duration"])
        hypno[s:min(e, total_sec)] = st

    is_sleep = (hypno < 0) & (~np.isnan(hypno))
    if not is_sleep.any():
        return None

    onsets = {name: find_onset(is_sleep, pmin, pfrac)
              for name, (pmin, pfrac) in ONSET_DEFS.items()}
    if onsets["OLD"] is None:
        return None

    # sleep fraction in the 10 min after each onset (the fake-onset detector)
    quality = {}
    for name, o in onsets.items():
        if o is None:
            quality[name] = float("nan")
        else:
            seg = is_sleep[o:o + 600]
            quality[name] = float(seg.mean()) if len(seg) else float("nan")

    win = int(WIN_SEC * sf); step = int(STEP_SEC * sf)
    out = {name: {"X": [], "y": [], "B": [], "n_eval": 0, "n_rej": 0,
                  "n_band": 0, "n_band_asleep": 0}
           for name in ONSET_DEFS}

    for start in range(0, len(signal) - win, step):
        center = (start + win / 2) / sf
        c_sec  = int(center)

        roles = {name: (None if o is None else role_of((center - o) / 60.0))
                 for name, o in onsets.items()}
        if all(r is None for r in roles.values()):
            continue

        # count how many windows fall in an evaluated band BEFORE the wake
        # filter, so we can see how many a later onset loses to scored sleep
        for name, r in roles.items():
            if r in ("target", "nontarget"):
                out[name]["n_band"] += 1

        # all bands must be scored WAKE: we separate two AWAKE states, and the
        # baseline must be plain quiet wake too
        if c_sec >= len(hypno) or hypno[c_sec] != 0:
            for name, r in roles.items():
                if r in ("target", "nontarget"):
                    out[name]["n_band_asleep"] += 1
            continue

        w = signal[start:start + win]
        bad = is_artifact(w)
        for name, r in roles.items():
            if r in ("target", "nontarget"):
                out[name]["n_eval"] += 1
                if bad:
                    out[name]["n_rej"] += 1
        if bad:
            continue

        s = spectrum(w, sf)               # computed ONCE, shared by all defs
        for name, r in roles.items():
            if r == "target":
                out[name]["X"].append(s); out[name]["y"].append(1)
            elif r == "nontarget":
                out[name]["X"].append(s); out[name]["y"].append(0)
            elif r == "baseline":
                out[name]["B"].append(s)

    nbins = len(freq_axis(sf))
    recs = {}
    for name, d in out.items():
        recs[name] = {
            "X": np.array(d["X"]) if d["X"] else np.zeros((0, nbins)),
            "y": np.array(d["y"]) if d["y"] else np.zeros(0, int),
            "B": np.array(d["B"]) if d["B"] else np.zeros((0, nbins)),
            "rej_pct": 100.0 * d["n_rej"] / max(d["n_eval"], 1),
            "lost_to_sleep_pct": 100.0 * d["n_band_asleep"] / max(d["n_band"], 1),
            "onset_min": np.nan if onsets[name] is None else onsets[name] / 60.0,
            "quality": quality[name],
        }
    return recs

# ----------------------------------------------------------------------
# Extraction with cache
# ----------------------------------------------------------------------
def extract_all(pairs):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            all_recs = pickle.load(fh)
        print(f"Loaded cached features from {CACHE_FILE} "
              f"(delete it to force re-extraction)\n")
        return all_recs

    all_recs = {name: [] for name in ONSET_DEFS}
    for psg, hyp in pairs:
        try:
            recs = process_recording(download(psg), download(hyp))
        except Exception as ex:
            print(f"{psg:<16s} ERROR {ex}")
            continue
        if recs is None:
            print(f"{psg:<16s} unusable, skipped")
            continue
        for name in ONSET_DEFS:
            recs[name]["name"] = psg
            recs[name]["subject"] = subject_id(psg)
            all_recs[name].append(recs[name])
        print(f"  extracted {psg}")
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(all_recs, fh)
    print(f"\nCached features to {CACHE_FILE}\n")
    return all_recs

# ----------------------------------------------------------------------
# Calibration (identical to v8)
# ----------------------------------------------------------------------
def calibrate(rec):
    """z-score this recording's evaluated spectra against its OWN baseline
    segment.  Uses no labels and no evaluated window."""
    mu = rec["B"].mean(0)
    sd = rec["B"].std(0) + 1e-6
    return (rec["X"] - mu) / sd

# ----------------------------------------------------------------------
# Evaluation (identical machinery to v8)
# ----------------------------------------------------------------------
def loso(recs, feat_key, subjects):
    rows = []
    for held in sorted(set(subjects)):
        tr = [i for i, s in enumerate(subjects) if s != held]
        te = [i for i, s in enumerate(subjects) if s == held]
        Xtr = np.vstack([recs[i][feat_key] for i in tr])
        ytr = np.concatenate([recs[i]["y"] for i in tr])
        Xte = np.vstack([recs[i][feat_key] for i in te])
        yte = np.concatenate([recs[i]["y"] for i in te])

        scaler = StandardScaler().fit(Xtr)                 # TRAIN ONLY
        pca    = PCA(n_components=N_PCA).fit(scaler.transform(Xtr))
        clf    = LogisticRegression(max_iter=1000).fit(
                     pca.transform(scaler.transform(Xtr)), ytr)
        Zte    = pca.transform(scaler.transform(Xte))
        rows.append({
            "subject": held, "n": len(yte),
            "bacc": balanced_accuracy_score(yte, clf.predict(Zte)),
            "auc": roc_auc_score(yte, clf.predict_proba(Zte)[:, 1])
                   if len(np.unique(yte)) > 1 else float("nan"),
        })
    return rows

def loso_ratio(recs, subjects, faxis):
    a = (faxis >= 8) & (faxis <= 12)
    t = (faxis >= 4) & (faxis <= 8)
    rows = []
    for held in sorted(set(subjects)):
        tr = [i for i, s in enumerate(subjects) if s != held]
        te = [i for i, s in enumerate(subjects) if s == held]
        Xtr = np.vstack([recs[i]["X"] for i in tr]); ytr = np.concatenate([recs[i]["y"] for i in tr])
        Xte = np.vstack([recs[i]["X"] for i in te]); yte = np.concatenate([recs[i]["y"] for i in te])
        rtr = (Xtr[:, a].mean(1) - Xtr[:, t].mean(1)).reshape(-1, 1)
        rte = (Xte[:, a].mean(1) - Xte[:, t].mean(1)).reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000).fit(rtr, ytr)
        rows.append({
            "subject": held, "n": len(yte),
            "bacc": balanced_accuracy_score(yte, clf.predict(rte)),
            "auc": roc_auc_score(yte, clf.predict_proba(rte)[:, 1])
                   if len(np.unique(yte)) > 1 else float("nan"),
        })
    return rows

def mean_of(rows, key):
    return np.nanmean([r[key] for r in rows])

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    files = local_file_list() or get_file_list()
    pairs = pair_psg_hypno(files)[:N_RECORDINGS]
    print(f"Processing {len(pairs)} recordings "
          f"({len(set(subject_id(p) for p, _ in pairs))} real subjects)\n")

    all_recs = extract_all(pairs)
    names = list(ONSET_DEFS)

    # ---------- table 1: where does onset land, and what survives ----------
    print("Per definition: onset (min into recording), q = sleep fraction in the")
    print("10 min after onset, tgt/ntg = surviving labelled windows, lost = % of")
    print("windows in the -10..0 bands that were thrown out for being scored sleep.\n")
    hdr = f"{'recording':<16s} {'subj':>4s}"
    for n in names:
        hdr += f" | {n:>9s} {'q':>4s} {'tgt':>4s} {'ntg':>4s} {'lost%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    by_name = {n: {r["name"]: r for r in all_recs[n]} for n in names}
    for psg, _ in pairs:
        if psg not in by_name[names[0]]:
            continue
        line = f"{psg:<16s} {subject_id(psg):>4s}"
        for n in names:
            r = by_name[n][psg]
            om = r["onset_min"]
            if not np.isfinite(om):
                line += f" | {'none':>9s} {'-':>4s} {'-':>4s} {'-':>4s} {'-':>6s}"
            else:
                line += (f" | {om:>9.1f} {r['quality']:>4.2f} "
                         f"{int(r['y'].sum()):>4d} {int((r['y']==0).sum()):>4d} "
                         f"{r['lost_to_sleep_pct']:>6.1f}")
        print(line)

    # ---------- inclusion rules, applied PER DEFINITION ----------
    kept = {}
    print()
    for n in names:
        k = []
        for r in all_recs[n]:
            why = None
            if not np.isfinite(r["onset_min"]):
                why = "no onset satisfies this definition"
            elif r["rej_pct"] > REJECT_MAX_PCT:
                why = f"artifact rejection {r['rej_pct']:.1f}% > {REJECT_MAX_PCT}%"
            elif r["y"].sum() < 5 or (r["y"] == 0).sum() < 5:
                why = (f"too few windows in one state (target={int(r['y'].sum())}, "
                       f"nontarget={int((r['y']==0).sum())})")
            elif len(r["B"]) < MIN_BASELINE_WINDOWS:
                why = f"only {len(r['B'])} baseline windows (< {MIN_BASELINE_WINDOWS})"
            if why:
                print(f"  [{n}] DROPPED {r['name']}: {why}")
            else:
                k.append(r)
        kept[n] = k
    print()
    for n in names:
        subs = sorted(set(r["subject"] for r in kept[n]))
        print(f"  {n:<10s} kept {len(kept[n]):>2d} recordings / {len(subs):>2d} subjects")

    common = set.intersection(*[set(r["subject"] for r in kept[n]) for n in names])
    print(f"\n  Common subject set (fair comparison): {len(common)} subjects: "
          f"{' '.join(sorted(common))}\n")

    faxis = freq_axis()
    results = {}

    print("=" * 80)
    print("Leave-one-SUBJECT-out, COMMON subject set (apples-to-apples)")
    print("=" * 80)
    print(f"  {'onset def':<11s} {'n subj':>6s} | {'ratio bacc':>10s} {'ratio auc':>9s}"
          f" | {'RAW bacc':>8s} {'RAW auc':>7s} | {'CAL bacc':>8s} {'CAL auc':>7s}")
    print("  " + "-" * 76)
    for n in names:
        recs = [r for r in kept[n] if r["subject"] in common]
        subs = [r["subject"] for r in recs]
        for r in recs:
            r["Xcal"] = calibrate(r)
        rr = loso(recs, "X", subs)
        rc = loso(recs, "Xcal", subs)
        rt = loso_ratio(recs, subs, faxis)
        results[n] = {"raw": rr, "cal": rc, "ratio": rt, "recs": recs}
        print(f"  {n:<11s} {len(set(subs)):>6d} |"
              f" {mean_of(rt,'bacc'):>10.3f} {mean_of(rt,'auc'):>9.3f} |"
              f" {mean_of(rr,'bacc'):>8.3f} {mean_of(rr,'auc'):>7.3f} |"
              f" {mean_of(rc,'bacc'):>8.3f} {mean_of(rc,'auc'):>7.3f}")

    # ---------- per-subject detail (bacc AND auc) ----------
    for metric in ["bacc", "auc"]:
        print(f"\nPer-subject {metric} on the common set (CAL features):")
        print("  " + f"{'subj':>4s}" + "".join(f" {n:>10s}" for n in names))
        print("  " + "-" * (4 + 11 * len(names)))
        tbl = {n: {r["subject"]: r[metric] for r in results[n]["cal"]} for n in names}
        for s in sorted(common):
            print("  " + f"{s:>4s}" + "".join(f" {tbl[n][s]:>10.3f}" for n in names))
        print("  " + f"{'mean':>4s}" +
              "".join(f" {np.nanmean(list(tbl[n].values())):>10.3f}" for n in names))

    # ---------- is the difference real, or 10 noisy subjects? ----------
    print("\nPaired comparison across the 10 subjects (CAL features), "
          "Wilcoxon signed-rank:")
    for metric in ["bacc", "auc"]:
        base = {r["subject"]: r[metric] for r in results["OLD"]["cal"]}
        for n in names[1:]:
            new = {r["subject"]: r[metric] for r in results[n]["cal"]}
            d = np.array([new[s] - base[s] for s in sorted(common)])
            try:
                stat, p = wilcoxon(d)
                ptxt = f"p = {p:.3f}"
            except ValueError:
                ptxt = "p = n/a (all differences zero)"
            print(f"  {metric}: {n} - OLD  mean diff = {d.mean():+.3f}  "
                  f"median = {np.median(d):+.3f}  {ptxt}  "
                  f"({int((d > 0).sum())}/{len(d)} subjects improved)")

    # ---------- does onset quality still predict who is separable? ----------
    print("\nOnset quality vs per-subject AUC (Spearman, n = %d subjects):" % len(common))
    for n in names:
        q, a = [], []
        tbl = {r["subject"]: r["auc"] for r in results[n]["cal"]}
        for s in sorted(common):
            q.append(np.mean([r["quality"] for r in results[n]["recs"]
                              if r["subject"] == s]))
            a.append(tbl[s])
        if np.std(q) < 1e-9:
            print(f"  {n:<11s} every onset is now equally clean (q = {q[0]:.2f}); "
                  f"the correlation is undefined, which is the point of the fix")
        else:
            rho, p = spearmanr(q, a)
            print(f"  {n:<11s} rho = {rho:+.3f}  p = {p:.3f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 5.0))
    for ax, n in zip(np.atleast_1d(axes), names):
        recs = results[n]["recs"]
        Xall = np.vstack([r["Xcal"] for r in recs])
        yall = np.concatenate([r["y"] for r in recs])
        Z = PCA(n_components=2).fit_transform(
                StandardScaler().fit_transform(Xall))
        ax.scatter(Z[yall == 0, 0], Z[yall == 0, 1], s=5, alpha=0.35,
                   color="#BA7517", label="Alert wake (-10..-5 min)")
        ax.scatter(Z[yall == 1, 0], Z[yall == 1, 1], s=5, alpha=0.35,
                   color="#185FA5", label="About to sleep (-3..0 min)")
        ax.set_title(f"{n}\nbacc {mean_of(results[n]['cal'],'bacc'):.3f}   "
                     f"AUC {mean_of(results[n]['cal'],'auc'):.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Effect of the sleep-onset definition on state separability "
                 "(common subject set, calibrated features)")
    plt.tight_layout()
    plt.savefig("onset_defs_v9.png", dpi=130)
    print("\nSaved figure: onset_defs_v9.png")

if __name__ == "__main__":
    main()
