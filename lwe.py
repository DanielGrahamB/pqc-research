# pyrefly: ignore [missing-import]
import math
import os
import torch


class LWEEncryptor:
    """
    Lindner-Peikert (LP11) LWE encryption with a bounded (mod-q) ciphertext.

    Why this and not GGH at the physical layer:
        A GGH ciphertext C = S @ B.T + E has no modulus, so its size grows with
        how "bad" the public basis B is (about 1e12 x the power of a 16-QAM
        symbol for the current keys). After normalising the transmit power the
        receiver needs ~110-120 dB SNR. Here every ciphertext coefficient lives
        in [-q/2, q/2), so the transmit power is fixed (q^2/12 per real
        dimension) whatever the security level. Channel noise simply adds to the
        LWE noise, and the mod-q decryption undoes any wrap-around (the same
        idea as Tomlinson-Harashima precoding).

    Keys (receiver = base station):
        s  : secret,  n x L,  small Gaussian
        A  : public,  n x n,  uniform mod q
        P  : public,  n x L,  P = r1 - A s  (mod q)

    Encryption of a message block m (length L, values 0..2^bits-1):
        c1 = e1 A + e2              (mod q)   message-independent
        c2 = e1 P + e3 + m * q/2^bits (mod q)

    Only c2 goes over the air. c1 depends only on the random e1, e2, which are
    drawn from a secret per-session seed (in a full system: derived from an
    ML-KEM shared secret). The receiver rebuilds c1 from the seed. Two receiver
    modes:
        "secret_key" : v = c2 + c1 s = m*q/2^bits + (e1 r1 + e2 s + e3) + noise
                       (true LWE decryption; small LWE-noise floor, LDPC removes it)
        "exact_mask" : v = c2 - (e1 P + e3) = m*q/2^bits + noise
                       (the seed alone strips the mask; no LWE noise)
    Eve has neither the seed nor s, so to her c2 looks uniform mod q.

    The seed MUST be secret. If it were public (as suggested in DeepJSCEC,
    Tung & Gunduz 2022), Eve could rebuild e1, e3 and remove the mask herself.
    """

    def __init__(self, n=192, L=384, q=4093, s_param=8.87, seed=None, device="cpu"):
        self.n, self.L, self.q = n, L, q
        self.sd = s_param / math.sqrt(2 * math.pi)       # Gaussian std from LP parameter s
        self.device = device
        # key generation (receiver side)
        g = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") if seed is None else seed)
        self.A = torch.randint(0, q, (n, n), generator=g, dtype=torch.float64)
        self.S = self._gauss((n, L), g)
        r1 = self._gauss((n, L), g)
        self.P = torch.remainder(r1 - self.A @ self.S, q)
        # per-session secret seed shared by UE and BS (stand-in for ML-KEM output)
        self.session_seed = int.from_bytes(os.urandom(8), "little")

    # ---------- helpers ----------
    def _gauss(self, shape, g):
        return torch.round(torch.randn(shape, generator=g, dtype=torch.float64) * self.sd)

    def centre(self, x):
        """Map values mod q to the centred range [-q/2, q/2)."""
        return torch.remainder(x + self.q // 2, self.q) - self.q // 2

    @property
    def lwe_noise_var(self):
        """Variance of e1 r1 + e2 s + e3 (receiver's LWE noise in 'secret_key' mode)."""
        return 2 * self.n * self.sd ** 4 + self.sd ** 2

    @property
    def coeff_std(self):
        """Std of one centred ciphertext coefficient (uniform on [-q/2, q/2))."""
        return self.q / math.sqrt(12.0)

    # ---------- encryption ----------
    def encrypt(self, m, bits=1, generator=None):
        """
        m: [N, L] integer tensor with values in 0 .. 2^bits - 1.
        Returns (c2, c1, mask), all [N, *] float64 mod q.
        c1 and mask are what the receiver rebuilds from the shared session seed.
        """
        N = m.shape[0]
        g = generator
        if g is None:
            g = torch.Generator().manual_seed(self.session_seed)
        e1 = self._gauss((N, self.n), g)
        e2 = self._gauss((N, self.n), g)
        e3 = self._gauss((N, self.L), g)
        dev = m.device
        A, P = self.A.to(dev), self.P.to(dev)
        e1, e2, e3 = e1.to(dev), e2.to(dev), e3.to(dev)
        c1 = torch.remainder(e1 @ A + e2, self.q)
        mask = torch.remainder(e1 @ P + e3, self.q)
        c2 = torch.remainder(mask + m.to(torch.float64) * (self.q / 2 ** bits), self.q)
        return c2, c1, mask

    # ---------- decryption ----------
    def unmask(self, c2_rx, c1=None, mask=None, mode="secret_key"):
        """Remove the mask from received (noisy) c2. Returns v in [0, q)."""
        if mode == "secret_key":
            v = c2_rx + c1 @ self.S.to(c2_rx.device)
        elif mode == "exact_mask":
            v = c2_rx - mask
        elif mode == "eve":          # no seed, no secret key: best guess is "mask = 0"
            v = c2_rx
        else:
            raise ValueError(mode)
        return torch.remainder(v, self.q)

    def llr(self, v, noise_var, bits=1):
        """
        Max-log LLRs (Sionna sign convention: log p(b=1)/p(b=0)) for the
        2^bits message points k*q/2^bits on the mod-q circle. Gray labelling.
        v: [N, L] in [0, q).  noise_var: scalar or broadcastable, per coefficient.
        Returns [N, L*bits].
        """
        M = 2 ** bits
        pts = torch.arange(M, dtype=torch.float64, device=v.device) * (self.q / M)
        d = self.centre(v.unsqueeze(-1) - pts)                  # circular distance
        metric = -(d ** 2) / (2 * noise_var.unsqueeze(-1) if torch.is_tensor(noise_var) and noise_var.dim() else 2 * noise_var)
        labels = gray_labels(bits).to(v.device)                 # [M, bits]
        out = []
        for b in range(bits):
            m1 = metric.masked_fill(~labels[:, b].bool(), -float("inf")).amax(-1)
            m0 = metric.masked_fill(labels[:, b].bool(), -float("inf")).amax(-1)
            out.append(m1 - m0)
        return torch.stack(out, -1).reshape(*v.shape[:-1], -1)


def gray_labels(bits):
    """Gray labels for points 0..2^bits-1 as a [2^bits, bits] 0/1 tensor (MSB first)."""
    M = 2 ** bits
    g = [k ^ (k >> 1) for k in range(M)]
    return torch.tensor([[(v >> (bits - 1 - b)) & 1 for b in range(bits)] for v in g], dtype=torch.float64)


def bits_to_symbols(b, bits):
    """[..., L*bits] 0/1 -> [..., L] integers using Gray labelling (inverse of gray_labels)."""
    b = b.reshape(*b.shape[:-1], -1, bits).to(torch.int64)
    val = torch.zeros(b.shape[:-1], dtype=torch.int64, device=b.device)
    for i in range(bits):
        val = val * 2 + b[..., i]
    # Gray -> binary index
    idx = val.clone()
    shift = val >> 1
    while bool(shift.any()):
        idx ^= shift
        shift >>= 1
    return idx
