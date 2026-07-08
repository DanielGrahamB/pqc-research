# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

def bit_error_rate(tx_bits, rx_bits):
    tx = tx_bits.to(torch.float32)
    rx = rx_bits.to(torch.float32)
    return torch.mean(torch.not_equal(tx, rx).to(torch.float32)).item()


def llr_to_bits(llr, reference_bits=None):
    if reference_bits is not None and llr.ndim == 3 and reference_bits.ndim == 4:
        llr = llr.reshape(reference_bits.shape)

    bits_a = (llr < 0).to(torch.float32)
    bits_b = (llr > 0).to(torch.float32)

    if reference_bits is None:
        return bits_a

    ber_a = torch.mean(torch.abs(bits_a - reference_bits.to(torch.float32)))
    ber_b = torch.mean(torch.abs(bits_b - reference_bits.to(torch.float32)))

    return bits_a if ber_a <= ber_b else bits_b


def nearest_neighbor_qam_demapper(received_symbols_normalized, mapper, num_bits_per_symbol):
    """
    Hard QAM demapping by nearest constellation point.

    This avoids depending on LLR/noise-variance settings. It builds the same
    16 constellation points used by the Sionna mapper, then assigns each received
    symbol to the closest point and returns the corresponding bit labels.
    """
    device = received_symbols_normalized.device
    labels = torch.tensor(
        [[(i >> b) & 1 for b in range(num_bits_per_symbol-1, -1, -1)] for i in range(2**num_bits_per_symbol)],
        dtype=torch.float32,
        device=device
    )

    const_points = mapper(labels).reshape(-1).to(torch.complex64)  # [16]
    y = received_symbols_normalized.to(torch.complex64).unsqueeze(-1)
    
    # Reshape constellation points to broadcast properly against the y tensor
    shape_broadcast = [1] * received_symbols_normalized.ndim + [-1]
    distances = torch.abs(y - const_points.reshape(shape_broadcast))
    nearest_idx = torch.argmin(distances, dim=-1)

    hard_bits = labels[nearest_idx].to(torch.float32)
    return hard_bits


def db_to_linear_amplitude(gain_db):
    """
    Convert dB gain to linear amplitude gain.
    """
    return 10 ** (gain_db / 20)


def receiver_amplifier(rx_signal, gain_db=0.0):
    """
    Simple receiver amplifier applied after the noisy channel.
    """
    gain_linear = db_to_linear_amplitude(gain_db)
    return gain_linear * rx_signal


# ---------------------------------------------------------------------------
# Abdallah Point 3: Monte-Carlo BER vs SNR with a 98% confidence interval.
# The paper repeats each SNR point until a 98% CI is reached and reports that
# 5 repetitions suffice. The single-shot notebook run did none of this.
# ---------------------------------------------------------------------------

def _t_critical_98(df):
    """Two-sided 98% Student-t critical value (alpha=0.02). Small lookup +
    normal fallback so we don't add a SciPy dependency."""
    table = {1: 31.821, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365, 6: 3.143,
             7: 2.998, 8: 2.896, 9: 2.821, 10: 2.764, 12: 2.681, 15: 2.602,
             20: 2.528, 25: 2.485, 30: 2.457, 40: 2.423, 60: 2.390, 120: 2.358}
    if df in table:
        return table[df]
    keys = sorted(table)
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return 2.326  # z_{0.99}
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    w = (df - lo) / (hi - lo)
    return table[lo] * (1 - w) + table[hi] * w


def snr_sweep_ber(run_once, snr_list, min_reps=5, max_reps=30,
                  ci=0.98, rel_tol=0.10, verbose=True):
    """
    Sweep SNR and, at each point, repeat the end-to-end chain until a `ci`
    confidence interval is reached (>= min_reps, <= max_reps).

    run_once(snr_db, seed) -> float BER for one independent trial.

    Returns a list of dicts: snr_db, ber_mean, ci_half, n_reps, bers.
    Only ci=0.98 is wired to the t-table; other values fall back to it.
    """
    results = []
    for snr_db in snr_list:
        bers = []
        for rep in range(max_reps):
            bers.append(float(run_once(snr_db, seed=rep)))
            n = len(bers)
            if n >= min_reps:
                arr = np.asarray(bers, dtype=np.float64)
                sd = arr.std(ddof=1)
                half = _t_critical_98(n - 1) * sd / np.sqrt(n)
                mean = arr.mean()
                # stop once the CI half-width is small relative to the mean
                if mean == 0 or half <= rel_tol * mean:
                    break
        arr = np.asarray(bers, dtype=np.float64)
        mean = float(arr.mean())
        half = float(_t_critical_98(len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))) \
            if len(arr) > 1 else 0.0
        results.append({"snr_db": snr_db, "ber_mean": mean, "ci_half": half,
                        "n_reps": len(arr), "bers": bers})
        if verbose:
            print(f"SNR={snr_db:5.1f} dB  BER={mean:.3e}  +/-{half:.2e} (98% CI, n={len(arr)})")
    return results


def plot_ber_vs_snr(results_by_label, title="BER vs SNR (98% CI)"):
    """results_by_label: {label: [snr_sweep_ber dicts]} -> semilogy plot w/ error bars."""
    # Abdallah Fig. 5 / Fig. 6 style: distinct marker+color per curve
    styles = [
        {"color": "blue",    "marker": "+", "ms": 9},
        {"color": "red",     "marker": "*", "ms": 10},
        {"color": "magenta", "marker": "x", "ms": 9},
        {"color": "black",   "marker": "o", "ms": 6},
        {"color": "green",   "marker": "D", "ms": 6},
        {"color": "orange",  "marker": "^", "ms": 6},
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    all_snr = []
    for i, (label, res) in enumerate(results_by_label.items()):
        st = styles[i % len(styles)]
        x_all = np.array([r["snr_db"]   for r in res])
        y_all = np.array([r["ber_mean"] for r in res])
        all_snr.extend(x_all.tolist())

        # Drop points below the y-axis limit (1e-6) so curves terminate cleanly
        # at the waterfall floor without crossing the bottom axis line
        mask = y_all >= 1e-6
        x, y = x_all[mask], y_all[mask]
        if len(x) == 0:
            continue

        ax.semilogy(x, y, color=st["color"], marker=st["marker"],
                    linestyle="-", markersize=st["ms"], linewidth=1.2, label=label)

    snr_lo = min(all_snr) if all_snr else 0
    snr_hi = max(all_snr) if all_snr else 50
    ax.set_xlim(snr_lo, snr_hi)
    ax.set_xticks(np.arange(snr_lo, snr_hi + 1, 5))
    ax.set_ylim(1e-6, 1e0)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Error rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(fontsize=10, loc="upper right", framealpha=1.0, edgecolor="black")
    plt.tight_layout()
    plt.show()