"""
Punctuated-Equilibrium Regime Detection for Crypto Prices
============================================================
Fits a Gaussian Hidden Markov Model (HMM) to log returns to identify
"stasis" (low-vol, range-bound) vs "punctuation" (high-vol, directional)
regimes -- the market-microstructure analogue of punctuated equilibrium.

USAGE
-----
1) With your own data:
   python regime_model.py --csv path/to/ohlcv.csv --price-col close

   CSV must have a column of prices (default column name: "close") and
   ideally a "date" or "timestamp" column (optional, used for x-axis).

2) Without data (demo mode, synthetic data that mimics punctuated
   equilibrium -- long calm stretches, sudden volatility bursts):
   python regime_model.py --demo

OUTPUTS
-------
- regime_summary.txt   : transition matrix, dwell times, state stats
- regime_plot.png      : price colored by regime + volatility w/ state probs
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simple_hmm import SimpleGaussianHMM as GaussianHMM


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------

def make_synthetic_punctuated_series(n=2000, seed=7):
    """
    Generate a price series that behaves like punctuated equilibrium:
    long low-vol stasis stretches, interrupted by short high-vol jumps,
    using a *true* underlying Markov chain so we can sanity-check the
    fitted model against ground truth.
    """
    rng = np.random.default_rng(seed)

    # True 2-state Markov chain: state 0 = stasis, state 1 = punctuation
    # Sticky stasis, short-lived punctuation -> classic PE signature
    P_true = np.array([
        [0.985, 0.015],   # from stasis
        [0.400, 0.600],   # from punctuation
    ])

    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice([0, 1], p=P_true[states[t - 1]])

    mu = {0: 0.0000, 1: 0.0015}       # slight drift during punctuations
    sigma = {0: 0.004, 1: 0.028}       # ~7x volatility jump

    returns = np.array([rng.normal(mu[s], sigma[s]) for s in states])
    price = 100 * np.exp(np.cumsum(returns))

    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({"date": dates, "close": price, "true_state": states})
    return df, P_true


def load_csv(path, price_col):
    df = pd.read_csv(path)
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found. Columns: {list(df.columns)}")
    date_col = None
    for c in ["date", "timestamp", "time", "Date", "Timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df["date"] = pd.to_datetime(df[date_col])
    else:
        df["date"] = pd.RangeIndex(len(df))
    df = df.rename(columns={price_col: "close"})
    return df[["date", "close"]].dropna().reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------

def build_features(df, vol_window=12):
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["realized_vol"] = df["log_ret"].rolling(vol_window).std()
    df = df.dropna().reset_index(drop=True)
    # Features fed to the HMM: return magnitude drives regime separation.
    # Using both return and rolling vol helps the model separate
    # "big single move" from "sustained turbulence".
    X = df[["log_ret", "realized_vol"]].values
    return df, X


# ----------------------------------------------------------------------
# 3. Fit HMM
# ----------------------------------------------------------------------

def fit_hmm(X, n_states=2, seed=42):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=seed,
    )
    model.fit(X)
    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)

    # Relabel states by volatility so state 0 = calmest ("stasis"),
    # last state = most volatile ("punctuation"). HMM label order is
    # arbitrary otherwise.
    vol_by_state = [X[hidden_states == s, 1].mean() for s in range(n_states)]
    order = np.argsort(vol_by_state)
    relabel = {old: new for new, old in enumerate(order)}
    hidden_states = np.array([relabel[s] for s in hidden_states])
    state_probs = state_probs[:, order]

    # Reorder transition matrix and means/covars to match relabeling
    P = model.transmat_[np.ix_(order, order)]
    means = model.means_[order]
    covars = model.covars_[order]

    return model, hidden_states, state_probs, P, means, covars


def dwell_time_stats(P):
    """Expected dwell time in state i under a Markov chain = 1 / (1 - P[i,i])."""
    return 1.0 / (1.0 - np.diag(P))


# ----------------------------------------------------------------------
# 4. Reporting
# ----------------------------------------------------------------------

def summarize(df, hidden_states, state_probs, P, means, covars, out_path):
    n_states = P.shape[0]
    names = ["stasis", "transition", "punctuation"][:n_states] if n_states <= 3 else \
            [f"state_{i}" for i in range(n_states)]

    dwell = dwell_time_stats(P)
    lines = []
    lines.append("PUNCTUATED-EQUILIBRIUM REGIME MODEL -- SUMMARY")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Transition matrix (rows = from, cols = to):")
    header = "        " + "  ".join(f"{n:>12s}" for n in names)
    lines.append(header)
    for i, row in enumerate(P):
        lines.append(f"{names[i]:>8s}" + "  ".join(f"{p:12.4f}" for p in row))
    lines.append("")
    lines.append("Expected dwell time (periods) per state -- 1/(1-P_ii):")
    for i, d in enumerate(dwell):
        lines.append(f"  {names[i]:>12s}: {d:8.1f} periods")
    lines.append("")
    lines.append("Emission stats per state (mean log-return, mean realized vol):")
    for i in range(n_states):
        lines.append(f"  {names[i]:>12s}: mean_ret={means[i][0]:+.5f}  vol_dim_mean={means[i][1]:.5f}")
    lines.append("")
    lines.append("Stationary distribution (long-run % of time in each state):")
    # solve for stationary dist: pi P = pi
    eigvals, eigvecs = np.linalg.eig(P.T)
    stat = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    stat = stat / stat.sum()
    for i, s in enumerate(stat):
        lines.append(f"  {names[i]:>12s}: {s*100:5.1f}%")
    lines.append("")
    current_state = hidden_states[-1]
    current_probs = state_probs[-1]
    lines.append(f"Most recent observation classified as: {names[current_state]}")
    lines.append("Current state probabilities: " +
                  ", ".join(f"{names[i]}={p:.3f}" for i, p in enumerate(current_probs)))
    if n_states >= 2:
        p_punct = current_probs[-1]
        lines.append("")
        lines.append(f"--> P(currently in/entering punctuation regime) = {p_punct:.3f}")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text


def plot_regimes(df, hidden_states, state_probs, out_path):
    n_states = state_probs.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

    ax = axes[0]
    x = df["date"].values
    y = df["close"].values
    for s in range(n_states):
        mask = hidden_states == s
        ax.scatter(x[mask], y[mask], s=6, color=colors[s], label=f"state {s}")
    ax.set_ylabel("Price")
    ax.set_title("Price colored by fitted regime (0 = calmest / stasis)")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    ax2.plot(x, df["realized_vol"].values, color="black", lw=0.8)
    ax2.set_ylabel("Realized vol")
    ax2.set_title("Rolling realized volatility")

    ax3 = axes[2]
    for s in range(n_states):
        ax3.plot(x, state_probs[:, s], label=f"P(state {s})", color=colors[s], lw=1)
    ax3.set_ylabel("State prob.")
    ax3.set_title("Smoothed regime probabilities")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV")
    ap.add_argument("--price-col", type=str, default="close")
    ap.add_argument("--n-states", type=int, default=2)
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    if args.csv:
        raw = load_csv(args.csv, args.price_col)
        true_states = None
    else:
        raw, P_true = make_synthetic_punctuated_series()
        true_states = raw["true_state"].values
        print("No --csv given: using synthetic punctuated-equilibrium demo data.")
        print("True generating transition matrix:\n", P_true)

    df, X = build_features(raw)
    model, hidden_states, state_probs, P, means, covars = fit_hmm(X, n_states=args.n_states)

    summary_path = f"{args.outdir}/regime_summary.txt"
    plot_path = f"{args.outdir}/regime_plot.png"

    text = summarize(df, hidden_states, state_probs, P, means, covars, summary_path)
    plot_regimes(df, hidden_states, state_probs, plot_path)

    print(text)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {plot_path}")

    if true_states is not None:
        # crude alignment check against ground truth (demo mode only)
        offset = len(raw) - len(df)
        aligned_true = true_states[offset:]
        acc = (aligned_true == hidden_states).mean()
        acc = max(acc, 1 - acc)  # label symmetry for 2-state case
        print(f"\n[demo mode] Recovered-state agreement with true simulated regime: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
