# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

def plot_constellation(tx, rx, title, limit=None, max_points=6000):
    tx = tx.detach().cpu().flatten()
    rx = rx.detach().cpu().flatten()

    if tx.numel() > max_points:
        tx = tx[:max_points]
    if rx.numel() > max_points:
        rx = rx[:max_points]

    plt.figure(figsize=(6, 6))
    plt.scatter(tx.real, tx.imag, s=8, label="Transmitted")
    plt.scatter(rx.real, rx.imag, s=8, alpha=0.45, label="Received")
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    if limit is not None:
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
    plt.legend()
    plt.show()


def plot_rx_only(rx, title, limit=None, max_points=6000):
    rx = rx.detach().cpu().flatten()

    if rx.numel() > max_points:
        rx = rx[:max_points]

    plt.figure(figsize=(6, 6))
    plt.scatter(rx.real, rx.imag, s=8, alpha=0.45)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    if limit is not None:
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
    plt.show()


def plot_signal_snapshot(tx_signal):
    """
    Time domain snapshot plot of the transmitted signal (real/imaginary parts).
    """
    x = tx_signal[0].detach().cpu()
    plt.figure(figsize=(12, 4))
    plt.plot(x.real, label="Real")
    plt.plot(x.imag, label="Imaginary")
    plt.title("OFDM transmitted signal snapshot")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_zoomed_signal(tx_signal, n_symbols=4, samples_per_symbol=80):
    """
    Zoomed step-plot and pulse-shaped smooth plot for the first n_symbols timestamps.
    """
    x = tx_signal[0].detach().cpu()
    x_np = x.numpy().flatten()
    x_narrow = x_np[:n_symbols]

    I = np.real(x_narrow)
    Q = np.imag(x_narrow)
    t_symbols = np.arange(n_symbols)

    # Print the values
    for k, sym in enumerate(x_narrow):
        print(f"t{k}: symbol = {sym:.4f}, I = {I[k]:.4f}, Q = {Q[k]:.4f}")

    # Plot 1: Step Plot
    plt.figure(figsize=(8, 4))
    plt.step(t_symbols, I, where="post", marker="o", label="I[n] = real part")
    plt.step(t_symbols, Q, where="post", marker="o", label="Q[n] = imaginary part")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(t_symbols, [f"t{k}" for k in t_symbols])
    # Approximate constellation scale levels to check visual
    plt.yticks([-0.9487, -0.3162, 0, 0.3162, 0.9487], labels=["-0.9487", "-0.3162", "0", "0.3162", "0.9487"])
    plt.title("Zoom: First 4 Generated OFDM Constellation Symbols")
    plt.xlabel("Symbol time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Pulse-Shaped Smoothed Plot
    I_rect = np.repeat(I, samples_per_symbol)
    Q_rect = np.repeat(Q, samples_per_symbol)
    filter_length = 41
    pulse = np.hanning(filter_length)
    pulse = pulse / np.sum(pulse)
    I_smooth = np.convolve(I_rect, pulse, mode="same")
    Q_smooth = np.convolve(Q_rect, pulse, mode="same")
    t = np.arange(len(I_smooth)) / samples_per_symbol

    plt.figure(figsize=(8, 4))
    plt.plot(t, I_smooth, label="I(t) smoothed")
    plt.plot(t, Q_smooth, label="Q(t) smoothed")
    for k in range(n_symbols):
        plt.axvline(k, linestyle="--", linewidth=1)
        plt.text(k, np.max(np.abs(I_smooth)) * 1.1, f"t{k}", ha="center")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Zoom: Pulse-Shaped View of First 4 OFDM Constellation Symbols")
    plt.xlabel("Time in symbol periods")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_noisy_time_domain(rx_signal, title="Noisy time-domain OFDM samples after attenuation + AWGN"):
    """
    Scatter plot of noisy time-domain samples after AWGN and attenuation.
    """
    y = rx_signal.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    plt.scatter(np.real(y).flatten(), np.imag(y).flatten(), s=5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, which="both", axis="both")
    plt.title(title)
    plt.show()
