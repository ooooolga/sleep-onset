import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert
from scipy.fft import next_fast_len
from scipy.stats import skew, kurtosis, wilcoxon
import mne
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import sleep_onset_def as v9        # file lists, onset rule, download helpers

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

CACHE_FILE = "features_v11.pkl"
ONSET_PERSIST_MIN, ONSET_PERSIST_FRAC = 10.0, 0.80     # PERSIST80, settled in v9

WIN_SEC   = 4.0
CTX_SEC   = 20.0            # long context for the instability features
STEP_SEC  = 2.0
ALPHA_BAND = (8.0, 12.0)
THETA_BAND = (4.0, 8.0)

INST_NAMES = ["a_logmean", "a_cv", "a_cross", "a_ac1", "a_frachigh",
              "a_skew", "a_bimod", "t_cv", "t_cross", "at_corr"]

Z_GRID = np.arange(-1.0, 4.01, 0.1)


# ----------------------------------------------------------------------
# Instability features from a pair of band envelopes
# ----------------------------------------------------------------------
def envelope_features(ea, et, sf):
    """ea, et: alpha and theta Hilbert envelopes over one window."""
    n = len(ea)
    dur = n / sf
    ma = ea.mean() + 1e-12
    med = np.median(ea)

    # median crossings: how often the envelope flips from below to above its
    # own middle.  This is the switching-rate proxy.
    above = ea > med
    a_cross = np.count_nonzero(np.diff(above)) / dur

    aboves_t = et > np.median(et)
    t_cross = np.count_nonzero(np.diff(aboves_t)) / dur

    # lag-1 autocorrelation of the envelope
    x = ea - ma
    denom = np.dot(x, x)
    a_ac1 = float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else 0.0

    sk = float(skew(ea))
    ku = float(kurtosis(ea, fisher=False))          # Pearson (normal == 3)
    a_bimod = (sk * sk + 1.0) / ku if ku > 1e-9 else 0.0

    # correlation between the two band envelopes within this window
    if ea.std() > 1e-12 and et.std() > 1e-12:
        at_corr = float(np.corrcoef(ea, et)[0, 1])
    else:
        at_corr = 0.0

    return np.array([
        np.log(ma),
        ea.std() / ma,
        a_cross,
        a_ac1,
        float(np.mean(ea > 1.5 * med)),
        sk,
        a_bimod,
        et.std() / (et.mean() + 1e-12),
        t_cross,
        at_corr,
    ], dtype=float)


def analytic_envelope(x):
    """Hilbert envelope, padded to a fast FFT length then trimmed back."""
    n = len(x)
    return np.abs(hilbert(x, N=next_fast_len(n)))[:n]


# ----------------------------------------------------------------------
# Per-recording extraction
# ----------------------------------------------------------------------
def process_recording(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if v9.CHANNEL not in raw.ch_names:
        return None
    raw.pick([v9.CHANNEL])
    sf = raw.info["sfreq"]
    raw.filter(l_freq=0.3, h_freq=45.0, verbose=False)
    signal = raw.get_data()[0] * 1e6

    # band-limited envelopes computed ONCE for the whole night, then sliced
    alpha = mne.filter.filter_data(signal, sf, ALPHA_BAND[0], ALPHA_BAND[1], verbose=False)
    theta = mne.filter.filter_data(signal, sf, THETA_BAND[0], THETA_BAND[1], verbose=False)
    env_a = analytic_envelope(alpha)
    env_t = analytic_envelope(theta)

    annot = mne.read_annotations(hyp_path)
    total_sec = int(len(signal) / sf)
    hypno = np.full(total_sec, np.nan)
    for a in annot:
        st = v9.STAGE_MAP.get(a["description"], np.nan)
        s = int(a["onset"]); e = int(a["onset"] + a["duration"])
        hypno[s:min(e, total_sec)] = st

    is_sleep = (hypno < 0) & (~np.isnan(hypno))
    if not is_sleep.any():
        return None
    onset = v9.find_onset(is_sleep, ONSET_PERSIST_MIN, ONSET_PERSIST_FRAC)
    if onset is None:
        return None
    seg = is_sleep[onset:onset + 600]
    quality = float(seg.mean()) if len(seg) else float("nan")

    is_wake = (hypno == 0)

    win  = int(WIN_SEC * sf)
    ctx  = int(CTX_SEC * sf)
    step = int(STEP_SEC * sf)
    half_extra = (ctx - win) // 2          # samples added on each side

    # band edges in minutes, used to keep the 20 s context inside its own band
    BANDS = {
        "target":    (v9.TARGET_FROM,    v9.TARGET_TO),
        "nontarget": (v9.NONTARGET_FROM, v9.NONTARGET_TO),
        "baseline":  (v9.BASELINE_FROM,  v9.BASELINE_TO),
    }

    store = {k: {"spec": [], "inst4": [], "inst20": [], "ok20": []}
             for k in BANDS}
    n_eval = n_rej = 0

    for start in range(0, len(signal) - win, step):
        center  = (start + win / 2) / sf
        rel_min = (center - onset) / 60.0
        role = v9.role_of(rel_min)
        if role is None:
            continue

        c_sec = int(center)
        if c_sec >= len(hypno) or hypno[c_sec] != 0:      # must be scored wake
            continue

        w = signal[start:start + win]
        if role in ("target", "nontarget"):
            n_eval += 1
        if v9.is_artifact(w):
            if role in ("target", "nontarget"):
                n_rej += 1
            continue

        # --- short-window features ---
        spec  = v9.spectrum(w, sf)
        inst4 = envelope_features(env_a[start:start + win],
                                  env_t[start:start + win], sf)

        # --- long-context features, only if the context is fully legal ---
        cs, ce = start - half_extra, start + win + half_extra
        ok20 = False
        inst20 = np.full(len(INST_NAMES), np.nan)
        if cs >= 0 and ce <= len(signal):
            # (a) entirely inside this window's own labelled band -> can never
            #     cross sleep onset or leak into the neighbouring state
            lo_min = (cs / sf - onset) / 60.0
            hi_min = (ce / sf - onset) / 60.0
            b0, b1 = BANDS[role]
            inside = (lo_min >= b0) and (hi_min <= b1)
            # (b) entirely scored wake, (c) entirely artifact-free
            if inside:
                s0, s1 = int(cs / sf), int(np.ceil(ce / sf))
                if s1 <= len(is_wake) and is_wake[s0:s1].all():
                    ctx_sig = signal[cs:ce]
                    if not v9.is_artifact(ctx_sig):
                        inst20 = envelope_features(env_a[cs:ce], env_t[cs:ce], sf)
                        ok20 = True

        store[role]["spec"].append(spec)
        store[role]["inst4"].append(inst4)
        store[role]["inst20"].append(inst20)
        store[role]["ok20"].append(ok20)

    def stack(role, key):
        v = store[role][key]
        return np.array(v) if v else np.zeros((0, len(INST_NAMES)))

    n_spec = len(v9.freq_axis(sf))
    spec_t, spec_n = store["target"]["spec"], store["nontarget"]["spec"]
    if not spec_t or not spec_n:
        return None

    X = np.vstack([np.array(spec_t), np.array(spec_n)])
    y = np.concatenate([np.ones(len(spec_t), int), np.zeros(len(spec_n), int)])
    I4 = np.vstack([stack("target", "inst4"), stack("nontarget", "inst4")])
    I20 = np.vstack([stack("target", "inst20"), stack("nontarget", "inst20")])
    ok = np.concatenate([np.array(store["target"]["ok20"], bool),
                         np.array(store["nontarget"]["ok20"], bool)])

    return {
        "spec": X, "inst4": I4, "inst20": I20, "ok20": ok, "y": y,
        "B_spec":  np.array(store["baseline"]["spec"])
                   if store["baseline"]["spec"] else np.zeros((0, n_spec)),
        "B_inst4": stack("baseline", "inst4"),
        "B_inst20": stack("baseline", "inst20"),
        "B_ok20":  np.array(store["baseline"]["ok20"], bool)
                   if store["baseline"]["ok20"] else np.zeros(0, bool),
        "rej_pct": 100.0 * n_rej / max(n_eval, 1),
        "onset_min": onset / 60.0,
        "quality": quality,
    }


def extract_all(pairs):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            recs = pickle.load(fh)
        print(f"Loaded cached features from {CACHE_FILE} "
              f"(delete it to force re-extraction)\n")
        return recs
    recs = []
    for psg, hyp in pairs:
        try:
            r = process_recording(v9.download(psg), v9.download(hyp))
        except Exception as ex:
            print(f"{psg:<16s} ERROR {ex}")
            continue
        if r is None:
            print(f"{psg:<16s} unusable, skipped")
            continue
        r["name"] = psg
        r["subject"] = v9.subject_id(psg)
        recs.append(r)
        print(f"  extracted {psg}")
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(recs, fh)
    print(f"\nCached features to {CACHE_FILE}\n")
    return recs


# ----------------------------------------------------------------------
# Building calibrated design matrices
# ----------------------------------------------------------------------
def build(rec, blocks):
    """Assemble the calibrated feature blocks named in `blocks`.

    Two restrictions, both applied to EVERY feature set so the comparison stays
    apples-to-apples:
      * evaluated rows are those where the 20 s context is valid;
      * the calibration reference is the baseline windows where the 20 s
        context is valid, for all blocks alike (including the spectrum).
    Returns (eval matrix, calibrated baseline matrix, y), or None if this
    recording no longer has enough baseline to calibrate against."""
    rows  = rec["ok20"]
    brows = rec["B_ok20"]
    if brows.sum() < v9.MIN_BASELINE_WINDOWS:
        return None
    parts, bparts = [], []
    for name in blocks:
        ev  = rec[name]
        ref = rec["B_" + name][brows]
        if not np.isfinite(ref).all():
            return None
        mu, sd = ref.mean(0), ref.std(0) + 1e-9
        parts.append((ev[rows] - mu) / sd)
        bparts.append((ref - mu) / sd)
    return np.hstack(parts), np.hstack(bparts), rec["y"][rows]


# ----------------------------------------------------------------------
# Model with an optional PCA applied to the spectrum block only
# ----------------------------------------------------------------------
class Pipe:
    """StandardScaler -> (PCA on the first `n_spec` columns) -> LogisticRegression.
    Instability columns are only 10-dimensional, so they are passed through
    without reduction; the spectrum is 60-dimensional and keeps the PCA
    compression used since v7, so results stay comparable across versions."""

    def __init__(self, n_spec, n_pca):
        self.n_spec, self.n_pca = n_spec, n_pca

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        if self.n_spec and self.n_pca:
            self.pca = PCA(n_components=self.n_pca).fit(Z[:, :self.n_spec])
        else:
            self.pca = None
        self.clf = LogisticRegression(max_iter=2000).fit(self._red(Z), y)
        return self

    def _red(self, Z):
        if self.pca is None:
            return Z
        return np.hstack([self.pca.transform(Z[:, :self.n_spec]),
                          Z[:, self.n_spec:]])

    def score(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return self.clf.decision_function(self._red(self.scaler.transform(X)))


# ----------------------------------------------------------------------
# Evaluation: leave-one-subject-out with a nested-CV personal threshold
# ----------------------------------------------------------------------
def base_z_predict(ev, base, c):
    if len(base) == 0:
        return (ev > 0).astype(int)
    return (((ev - base.mean()) / (base.std() + 1e-9)) > c).astype(int)


def choose_c(train, blocks, n_spec, n_pca):
    subs = sorted(set(r["subject"] for r in train))
    scored = []
    for t in subs:
        itr = [r for r in train if r["subject"] != t]
        ite = [r for r in train if r["subject"] == t]
        if len(set(r["subject"] for r in itr)) < 2:
            continue
        Xtr = np.vstack([r["_X"] for r in itr]); ytr = np.concatenate([r["_y"] for r in itr])
        m = Pipe(n_spec, n_pca).fit(Xtr, ytr)
        scored.append([(m.score(r["_X"]), m.score(r["_B"]), r["_y"]) for r in ite])
    best_c, best = Z_GRID[0], -np.inf
    for c in Z_GRID:
        vals = []
        for recs in scored:
            p = np.concatenate([base_z_predict(ev, b, c) for ev, b, _ in recs])
            yy = np.concatenate([y for _, _, y in recs])
            if len(np.unique(yy)) > 1:
                vals.append(balanced_accuracy_score(yy, p))
        if vals and np.mean(vals) > best:
            best, best_c = np.mean(vals), c
    return best_c


def evaluate(recs, blocks, n_spec, n_pca):
    """Attaches _X/_B/_y to each record for this feature set, then runs LOSO."""
    usable = []
    for r in recs:
        built = build(r, blocks)
        if built is None:
            continue
        X, B, y = built
        if len(np.unique(y)) < 2 or len(B) < v9.MIN_BASELINE_WINDOWS:
            continue
        r["_X"], r["_B"], r["_y"] = X, B, y
        usable.append(r)

    subs = sorted(set(r["subject"] for r in usable))
    auc, bacc_g, bacc_z, cs = {}, {}, {}, {}
    for held in subs:
        train = [r for r in usable if r["subject"] != held]
        test  = [r for r in usable if r["subject"] == held]
        c = choose_c(train, blocks, n_spec, n_pca)
        cs[held] = c
        Xtr = np.vstack([r["_X"] for r in train]); ytr = np.concatenate([r["_y"] for r in train])
        m = Pipe(n_spec, n_pca).fit(Xtr, ytr)
        scored = [(m.score(r["_X"]), m.score(r["_B"]), r["_y"]) for r in test]
        yy = np.concatenate([y for _, _, y in scored])
        ev = np.concatenate([e for e, _, _ in scored])
        auc[held]    = roc_auc_score(yy, ev) if len(np.unique(yy)) > 1 else np.nan
        bacc_g[held] = balanced_accuracy_score(yy, (ev > 0).astype(int))
        bacc_z[held] = balanced_accuracy_score(
            yy, np.concatenate([base_z_predict(e, b, c) for e, b, _ in scored]))
    return {"auc": auc, "global": bacc_g, "basez": bacc_z, "c": cs,
            "n_subj": len(subs)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    files = v9.local_file_list() or v9.get_file_list()
    pairs = v9.pair_psg_hypno(files)[:v9.N_RECORDINGS]
    recs = extract_all(pairs)

    # inclusion rules, unchanged
    kept = []
    for r in recs:
        why = None
        if r["rej_pct"] > v9.REJECT_MAX_PCT:
            why = f"artifact rejection {r['rej_pct']:.1f}% > {v9.REJECT_MAX_PCT}%"
        elif r["y"].sum() < 5 or (r["y"] == 0).sum() < 5:
            why = "too few windows in one state"
        elif len(r["B_spec"]) < v9.MIN_BASELINE_WINDOWS:
            why = f"only {len(r['B_spec'])} baseline windows"
        if why:
            print(f"  DROPPED {r['name']}: {why}")
        else:
            kept.append(r)

    subs = sorted(set(r["subject"] for r in kept))
    print(f"\nkept {len(kept)} recordings / {len(subs)} subjects\n")

    # how many windows survive the 20 s context requirement?
    tot = sum(len(r["ok20"]) for r in kept)
    ok  = sum(int(r["ok20"].sum()) for r in kept)
    btot = sum(len(r["B_ok20"]) for r in kept)
    bok = sum(int(r["B_ok20"].sum()) for r in kept)
    print(f"20 s context valid for {ok}/{tot} labelled windows ({100*ok/tot:.1f}%) "
          f"and {bok}/{btot} baseline windows ({100*bok/btot:.1f}%)")
    print("All feature sets below are evaluated on those same rows.\n")

    n_spec = kept[0]["spec"].shape[1]
    n_inst = len(INST_NAMES)

    SETS = [
        ("SPEC-PC2",            ["spec"],            n_spec, 2),
        ("SPEC-PC5  (control)", ["spec"],            n_spec, 5),
        ("INST4  alone",        ["inst4"],           0,      0),
        ("INST20 alone",        ["inst20"],          0,      0),
        ("SPEC-PC2 + INST4",    ["spec", "inst4"],   n_spec, 2),
        ("SPEC-PC2 + INST20",   ["spec", "inst20"],  n_spec, 2),
        ("SPEC-PC5 + INST20",   ["spec", "inst20"],  n_spec, 5),
    ]

    results = {}
    print("=" * 76)
    print("Leave-one-SUBJECT-out, identical rows, only the columns differ")
    print("=" * 76)
    print(f"  {'feature set':<22s} {'dims':>5s} | {'AUC':>6s} | "
          f"{'bacc GLOBAL':>11s} {'bacc BASE-Z':>11s}")
    print("  " + "-" * 62)
    for name, blocks, ns, npca in SETS:
        res = evaluate(kept, blocks, ns, npca)
        results[name] = res
        d = (npca if npca else 0) + (n_inst if any(b.startswith("inst") for b in blocks) else 0)
        print(f"  {name:<22s} {d:>5d} | {np.nanmean(list(res['auc'].values())):>6.3f} | "
              f"{np.nanmean(list(res['global'].values())):>11.3f} "
              f"{np.nanmean(list(res['basez'].values())):>11.3f}")

    # ---------- the decisive paired comparisons ----------
    print("\nPaired across the 10 subjects (Wilcoxon).  The question is whether")
    print("instability adds anything the spectrum did not already have:")
    def paired(a, b, metric):
        va = np.array([results[a][metric][s] for s in subs])
        vb = np.array([results[b][metric][s] for s in subs])
        d = va - vb
        try:
            _, p = wilcoxon(d); pt = f"p = {p:.3f}"
        except ValueError:
            pt = "p = n/a"
        print(f"  {metric:>6s}  {a:<22s} - {b:<22s} "
              f"{d.mean():+.3f}  {pt}  ({int((d>0).sum())}/{len(d)} up)")
    for metric in ["auc", "basez"]:
        paired("SPEC-PC2 + INST20", "SPEC-PC2", metric)
        paired("SPEC-PC2 + INST4",  "SPEC-PC2", metric)
        paired("SPEC-PC5  (control)", "SPEC-PC2", metric)   # the boring explanation

    # ---------- per-subject ----------
    print("\nPer-subject AUC:")
    names = [s[0] for s in SETS]
    print("  " + f"{'subj':>4s}" + "".join(f" {n.split()[0]:>10s}" for n in names))
    print("  " + "-" * (4 + 11 * len(names)))
    for s in subs:
        print("  " + f"{s:>4s}" + "".join(f" {results[n]['auc'][s]:>10.3f}" for n in names))
    print("  " + f"{'mean':>4s}" +
          "".join(f" {np.nanmean(list(results[n]['auc'].values())):>10.3f}" for n in names))

    # ---------- which instability features actually differ between states? ----------
    print("\nWhich instability features separate the two states at all?")
    print("(effect size = Cohen's d on baseline-calibrated INST20, pooled over")
    print(" all subjects; |d| < 0.2 is negligible)")
    Xs, ys = [], []
    for r in kept:
        b = build(r, ["inst20"])
        if b is None:
            continue
        Xs.append(b[0]); ys.append(b[2])
    Xc = np.vstack(Xs); yc = np.concatenate(ys)
    for i, nm in enumerate(INST_NAMES):
        t, n = Xc[yc == 1, i], Xc[yc == 0, i]
        sp = np.sqrt((t.var() + n.var()) / 2) + 1e-12
        d = (t.mean() - n.mean()) / sp
        flag = "  <-- " if abs(d) >= 0.2 else ""
        print(f"    {nm:<12s} d = {d:+.3f}{flag}")

    # ---------- subject 00 vs subject 05 spectra ----------
    plot_spectra(kept, ["00", "05"])

    # ---------- results figure ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    aucs = [np.nanmean(list(results[n]["auc"].values())) for n in names]
    bz   = [np.nanmean(list(results[n]["basez"].values())) for n in names]
    ax.bar(x - 0.2, aucs, 0.4, color="#185FA5", label="AUC")
    ax.bar(x + 0.2, bz,   0.4, color="#BA7517", label="balanced accuracy (BASE-Z)")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.4, 0.8); ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_title("Does alpha instability add anything the spectrum lacks?")
    plt.tight_layout()
    plt.savefig("instability_v11.png", dpi=130)
    print("\nSaved figure: instability_v11.png")


def plot_spectra(kept, want):
    """Mean log spectrum per state for a separable and a non-separable subject."""
    faxis = v9.freq_axis()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    diffs = {}
    for ax, sid in zip(axes[:2], want):
        rs = [r for r in kept if r["subject"] == sid]
        X = np.vstack([r["spec"] for r in rs]); y = np.concatenate([r["y"] for r in rs])
        mt, mn = X[y == 1].mean(0), X[y == 0].mean(0)
        diffs[sid] = mt - mn
        ax.plot(faxis, mn, color="#BA7517", lw=1.8, label="alert wake (-10..-5 min)")
        ax.plot(faxis, mt, color="#185FA5", lw=1.8, label="about to sleep (-3..0 min)")
        ax.axvspan(8, 12, color="grey", alpha=0.15)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("log power")
        ax.set_title(f"subject {sid}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for sid, col in zip(want, ["#185FA5", "#C0392B"]):
        axes[2].plot(faxis, diffs[sid], color=col, lw=1.8, label=f"subject {sid}")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].axvspan(8, 12, color="grey", alpha=0.15)
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("log power difference (about-to-sleep minus alert)")
    axes[2].set_title("what actually changes"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    fig.suptitle("A separable subject vs a non-separable one (grey band = alpha, 8-12 Hz)")
    plt.tight_layout()
    plt.savefig("spectra_00_vs_05_v11.png", dpi=130)
    print("\nSaved figure: spectra_00_vs_05_v11.png")


if __name__ == "__main__":
    main()
