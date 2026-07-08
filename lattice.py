# pyrefly: ignore [missing-import]
import torch
from metrics import generate_lattice_bases, hadamard_ratio, matrix_condition_number

class LatticeBasedEncryptor:
    def __init__(
        self,
        n,
        sigma=None,
        rounds=5,
        max_coeff=3,
        goodTh=0.8,
        badTh=0.001,
        max_bad_attempts=30,
        insecure_identity_R=False,
        pert=2,
        k=None,
        verbose=True,
    ):
        """
        n: vector dimension, should match fft_size (== lattice dimension == #sub-carriers)
        sigma: amplitude of the +/- error vector E (Abdallah Sec. 4.2.4).
               If None, it is auto-selected inside the valid window
                    1/(2*rho_B) < sigma < 1/(2*rho_R)
               (geometric-mean placement). The paper's "sigma a positive integer"
               is only admissible when rho_R < 0.5; that almost never holds together
               with HR(R) >= 0.8, so we honour the real correctness/security bound
               instead and warn if an explicit sigma falls outside the window.

        Destination user's lattice keypair:
            R = private good basis,  B = public degraded basis.
        """
        self.n = n

        self.R, self.B, self.info = generate_lattice_bases(
            n=n,
            rounds=rounds,
            max_coeff=max_coeff,
            goodTh=goodTh,
            badTh=badTh,
            max_bad_attempts=max_bad_attempts,
            insecure_identity_R=insecure_identity_R,
            pert=pert,
            k=k,
            verbose=verbose,
        )

        # pinv is safer than inv for noisy floating-point recovery.
        self.B_inv = torch.linalg.pinv(self.B.to(torch.complex128))
        self.R_inv = torch.linalg.pinv(self.R.to(torch.complex128))

        # --- Abdallah Point 1: select / validate sigma against the rule ---
        lo = self.info["sigma_min_secure"]   # 1/(2*rho_B): below this, eavesdropper reads it
        hi = self.info["sigma_max_correct"]  # 1/(2*rho_R): above this, legit decryption errors
        if sigma is None:
            if lo < hi:
                # Place sigma high in the window: as large as correctness allows
                # (maximally scrambles an eavesdropper) while staying < sigma_max.
                sigma = float(max(0.9 * hi, min(2.0 * lo, hi)))
            else:
                sigma = 0.5 * hi
                print("WARNING: empty sigma window; scheme is not simultaneously "
                      "correct and secure for this basis pair.")
        self.sigma = sigma

        if not (sigma < hi):
            print(f"WARNING: sigma={sigma:.3e} >= sigma_max_correct={hi:.3e}; "
                  "legitimate decryption will incur errors (Abdallah Sec. 4.2.4).")
        if not (sigma > lo):
            print(f"WARNING: sigma={sigma:.3e} <= sigma_min_secure={lo:.3e}; "
                  "an eavesdropper holding only B can recover the plaintext.")

        if verbose:
            print("Private basis R shape:", tuple(self.R.shape),
                  "| Public basis B shape:", tuple(self.B.shape))
            print(f"sigma = {self.sigma:.4e}  (window {lo:.3e} .. {hi:.3e})")
            if n < 350:
                print(f"NOTE: lattice dimension n={n} < 350. Abdallah Sec. 4.3 requires "
                      "n > 350 (>=400 to resist GGH/NTRU cryptanalysis) for security; "
                      "smaller n is for performance study only.")

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

        S = S.to(torch.complex128)

        error_sign = 2 * torch.randint(
            low=0,
            high=2,
            size=S.shape,
            device=S.device
        ) - 1

        E = self.sigma * error_sign.to(torch.complex128)
        C = torch.matmul(S, self.B.to(torch.complex128).T.to(S.device)) + E
        # Keep double precision to avoid precision limits in double-precision mod/demod
        return C.to(torch.complex128)

    def decrypt_linear(self, encrypted_rx_symbols):
        """
        Simple inverse recovery:
            S_hat = C @ B^{-T}
        """
        C_hat = encrypted_rx_symbols.to(torch.complex128)
        return torch.matmul(C_hat, self.B_inv.T.to(C_hat.device))

    def decrypt_w_babai(self, encrypted_rx_symbols, use_round=True):
        """
        Abdallah Algorithm 3 / Babai's ROUND-OFF decryption.

        Row-vector equivalent of the paper's   S_hat = B^{-1} R [R^{-1} C] :
            Y          = C_hat @ R_inv.T
            Y_integer  = round(Y)            # NEAREST integer
            lattice_pt = Y_integer @ R.T
            S_hat      = lattice_pt @ B_inv.T

        IMPORTANT (Abdallah Sec. 3.3.1 / 4.2.3): the paper prints the operator as
        the ceiling symbol but its own text says "round to the nearest integers",
        and Babai's round-off is defined with nearest-integer rounding. Using
        ceiling biases every coordinate by ~+0.5 and breaks recovery, so
        use_round=False is kept only as an explicit (discouraged) experiment.
        """
        C_hat = encrypted_rx_symbols.to(torch.complex128)
        device = C_hat.device

        R = self.R.to(torch.complex128).to(device)
        B_inv = self.B_inv.to(device)
        R_inv = self.R_inv.to(device)

        Y = torch.matmul(C_hat, R_inv.T)

        if use_round:
            Y_integer = torch.round(Y.real) + 1j * torch.round(Y.imag)
        else:
            print("WARNING: use_round=False uses ceiling, which is NOT Babai "
                  "round-off and will corrupt decryption (see docstring).")
            Y_integer = torch.ceil(Y.real) + 1j * torch.ceil(Y.imag)

        Y_integer = Y_integer.to(torch.complex128)
        lattice_point = torch.matmul(Y_integer, R.T)
        S_hat = torch.matmul(lattice_point, B_inv.T)
        return S_hat

    def eavesdropper_babai(self, encrypted_rx_symbols):
        """
        Best an eavesdropper can do with only the PUBLIC (bad) basis B:
        Babai round-off in B. If B is genuinely bad and sigma is in the valid
        window, this returns garbage -- which is the whole point of the scheme.
        Used by the security sweep to measure attacker symbol-error-rate.
        """
        C_hat = encrypted_rx_symbols.to(torch.complex128)
        device = C_hat.device
        B = self.B.to(torch.complex128).to(device)
        B_inv = self.B_inv.to(device)
        Y = torch.matmul(C_hat, B_inv.T)
        Y_integer = (torch.round(Y.real) + 1j * torch.round(Y.imag)).to(torch.complex64)
        # coordinates in the B basis are already the message coordinates S
        return Y_integer.to(torch.complex64)


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