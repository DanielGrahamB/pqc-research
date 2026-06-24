# pyrefly: ignore [missing-import]
import torch
from metrics import generate_lattice_bases, hadamard_ratio, matrix_condition_number

class LatticeBasedEncryptor:
    def __init__(
        self,
        n,
        sigma=0.001,
        num_ops=45,
        max_coeff=2,
        goodTh=0.8,
        badTh=0.01,
        max_bad_attempts=70,
        max_cond=1e5
    ):
        """
        n: vector dimension, should match fft_size
        sigma: small encryption error value

        Destination user's lattice keypair:
            R = private good basis
            B = public degraded basis
        """
        self.n = n
        self.sigma = sigma

        self.R, self.B = generate_lattice_bases(
            n=n,
            num_ops=num_ops,
            max_coeff=max_coeff,
            goodTh=goodTh,
            badTh=badTh,
            max_bad_attempts=max_bad_attempts,
            max_cond=max_cond
        )

        # pinv is safer than inv for noisy floating-point recovery.
        self.B_inv = torch.linalg.pinv(self.B)
        self.R_inv = torch.linalg.pinv(self.R)

        print("Private basis R shape:", self.R.shape)
        print("Public basis B shape:", self.B.shape)
        print("Hadamard ratio R:", hadamard_ratio(self.R.real).item())
        print("Hadamard ratio B:", hadamard_ratio(self.B.real).item())
        print("Condition number B:", matrix_condition_number(self.B.real))

    def encrypt(self, symbols_lattice):
        """
        Abdallah Algorithm 2 idea, row-vector version:
            C = S @ B.T + E

        Important:
            For Babai recovery, S should be integer-like QAM coordinates,
            e.g. 16-QAM levels {-3, -1, +1, +3}, not normalized Sionna points.
        """
        if symbols_lattice.dim() == 4 and symbols_lattice.shape[-1] == 1:
            S = symbols_lattice.squeeze(-1)
        else:
            S = symbols_lattice

        S = S.to(torch.complex64)

        error_sign = 2 * torch.randint(
            low=0,
            high=2,
            size=S.shape,
            device=S.device
        ) - 1

        E = self.sigma * error_sign.to(torch.complex64)
        C = torch.matmul(S, self.B.T.to(S.device)) + E
        return C

    def decrypt_linear(self, encrypted_rx_symbols):
        """
        Simple inverse recovery:
            S_hat = C @ B^{-T}
        """
        C_hat = encrypted_rx_symbols.to(torch.complex64)
        return torch.matmul(C_hat, self.B_inv.T.to(C_hat.device))

    def decrypt_w_babai(self, encrypted_rx_symbols, use_round=True):
        """
        Abdallah Algorithm 3 / Babai-like decryption.

        Row-vector equivalent of:
            S_hat = B^{-1} R [R^{-1} C]

        With row vectors:
            Y          = C_hat @ R_inv.T
            Y_integer  = round(Y) or ceil(Y)
            lattice_pt = Y_integer @ R.T
            S_hat      = lattice_pt @ B_inv.T
        """
        C_hat = encrypted_rx_symbols.to(torch.complex64)
        device = C_hat.device

        R = self.R.to(device)
        B_inv = self.B_inv.to(device)
        R_inv = self.R_inv.to(device)

        Y = torch.matmul(C_hat, R_inv.T)

        if use_round:
            Y_integer = torch.round(Y.real) + 1j * torch.round(Y.imag)
        else:
            Y_integer = torch.ceil(Y.real) + 1j * torch.ceil(Y.imag)

        Y_integer = Y_integer.to(torch.complex64)
        lattice_point = torch.matmul(Y_integer, R.T)
        S_hat = torch.matmul(lattice_point, B_inv.T)
        return S_hat


def qam16_lattice_to_normalized(symbols_lattice):
    """
    Convert integer-like 16-QAM lattice levels ±1, ±3 back to
    normalized unit-power 16-QAM levels by dividing by sqrt(10).
    """
    scale = torch.sqrt(torch.tensor(10.0, dtype=torch.float32, device=symbols_lattice.device))
    return symbols_lattice / scale


def qam16_slicer_lattice(symbols_lattice):
    """
    Slice raw Babai output to the nearest valid 16-QAM lattice point.
    Valid real/imag levels are {-3, -1, +1, +3}.
    """
    x = symbols_lattice.to(torch.complex64)
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=x.device)

    real_idx = torch.argmin(torch.abs(x.real.unsqueeze(-1) - levels), dim=-1)
    imag_idx = torch.argmin(torch.abs(x.imag.unsqueeze(-1) - levels), dim=-1)

    real_sliced = levels[real_idx]
    imag_sliced = levels[imag_idx]
    return (real_sliced + 1j * imag_sliced).to(torch.complex64)
