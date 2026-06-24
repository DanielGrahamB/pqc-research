# pyrefly: ignore [missing-import]
import torch

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
