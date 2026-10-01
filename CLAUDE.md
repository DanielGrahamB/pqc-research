# Handoff: sub-band lattice encryption over a MIMO uplink

Project folder: `/Users/boazdanielgraham/Documents/PYTHON/Wireless Simulation/ps-layer-security`
Main notebook: `PQC_MIMO_Simulations_subband.ipynb` (plus `lattice.py`, `metrics.py`, `ber.py`, `plots.py`, `test_image.png`)

---

## 1. Goal of the current experiment

Reproduce the image experiment of **Shankar & Mishra (2025, RLWE + LDPC image encryption)**, but with a more reliable design:

- **Sub-band lattice encryption + 5G LDPC**
- over a **realistic 3GPP MU-MIMO uplink** (Sionna, UMi / UMa / RMa, 4 users, 8 BS antennas, 128 subcarriers, 30 kHz, 16-QAM)
- measuring **reliability (image fidelity: BER, BLER, PSNR, SSIM)** and **confidentiality (Eve's BER, histograms, entropy)**.

Wording rule: in the paper, write "reliability (image fidelity) and confidentiality". Our scheme does **not** give cryptographic integrity (tamper detection). That would need a MAC or signature (e.g. ML-DSA), which is future work.

Writing style: simple, plain English; keep standard technical terms.

---

## 2. Bugs found and fixed (verify they are in the notebook)

### 2.1 Error vector was 3.16x too large (√10 scaling bug)
`lattice.py` `encrypt()` expects integer QAM levels (±1, ±3). The notebook passed **normalised** Sionna symbols, then multiplied by √10 at the receiver. That also scaled the error vector E by √10, pushing σ about 2.85x past the safe limit `1/(2·rho_R)`. Result: error floors, noisy images, big run-to-run variation.

Fix (3 places: `Model.call`, `HighFidelityImageModel.call_with_static_channel`, `SubbandModel`):
```python
qs = torch.sqrt(torch.tensor(10.0, dtype=torch.float64, device=x.device))
x_enc_blocks = self._lattice_engine.encrypt(x_reshaped.to(torch.complex128) * qs) / qs
```
In `SubbandModel`: `eng.encrypt(blk * 10**0.5) / 10**0.5` and the same in the power-calibration line.

### 2.2 LMMSE equalizer wipes out whole user streams
Ciphertext power is about **6×10¹²**; Sionna's LMMSE assumes unit power. For users with similar channels, leftover inter-user interference breaks Babai decoding for the whole stream (horizontal bands in Bob's image). Numpy test at 60 dB, user correlation 0.99: LMMSE gave 82% / 62% symbol errors on 2 users; ZF gave 0.

Fix: `from sionna.phy.ofdm import ZFEqualizer`, and in `Model.__init__` and `InterpolationModel.__init__` replace `LMMSEEqualizer(...)` with `ZFEqualizer(...)` (keep the attribute name `self._lmmse_equ`).

### 2.3 Different results from the "Premier" notebook
Same notebook code, but the Premier run most likely used an **older `lattice.py` / `metrics.py`** with much milder public-key mixing (small B entries, small ciphertext power), so LMMSE worked. Current `metrics.py` uses 5 full mixing rounds (max |B| ≈ 6.7×10⁶).
To confirm: print `bob_model._lattice_engine.B.abs().max()` and `getattr(engine, "info", "old lattice.py")` in both notebooks.

### 2.4 Reproducibility
Add at the top (after `import torch`):
```python
import random
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
```

---

## 3. Results so far (before all fixes were applied — rerun needed)

- Original full-band 128×4 (n=512) hits a BER floor ~1–2×10⁻³; LDPC barely helps (whole blocks fail).
- **Sub-band 16×12 (n=192) + LDPC** was the only config with no floor: BER 3.6e-4 (60 dB), 8.5e-5 (70 dB), 0 (80 dB).
- Same lattice size n=384: full-band 128×3 vs sub-band 32×12 were about equal. Likely because 128 subcarriers × 30 kHz = 3.84 MHz is only ~2 coherence bands in UMi, so the channel is not very frequency-selective. The 16×12 gain is probably mostly from the smaller n.
- Image figure is strong evidence of **error containment**: full-band errors = horizontal streaks; sub-band errors = small patches; sub-band 32×12 + LDPC = perfect image at 60–70 dB.
- 8×12 (n=96, 16 separate keys) was worst; probably bad keys + unnormalised power.

---

## 4. Key finding: the true SNR problem

The Eb/No axis ignores the ciphertext power (~10¹²), so "60 dB" in the plots is not a fair 60 dB. Numpy tests with the actual key generator (AWGN, transmit power normalised to 1):

| Scheme | HR(B) (secure if ≤ 0.001) | True SNR for clean decoding |
|---|---|---|
| GGH, current keys (5 rounds, coeff ≤ 3) | 4×10⁻⁴ (secure) | ~110–120 dB |
| GGH, 3 rounds, coeff 2 | 1×10⁻² | ~90 dB |
| GGH, 1–2 mild rounds | 0.3–0.5 (insecure) | ~40 dB |
| LWE (Lindner–Peikert n=192, q=4093), full ciphertext sent | secure | ~50 dB |
| LWE, `d` part pre-shared via secret seed | secure only if seed is secret | ~20 dB uncoded (BER 2e-2 at 10 dB, fixable by LDPC) |

Conclusion: **with secure keys, GGH at the physical layer cannot reach Shankar (0–12 dB Eb/No) or Jayasinghe (–2.5 to 15 dB) ranges.** More secure public key → bigger ciphertext → higher SNR needed. This security-vs-SNR trade-off is itself a reportable finding.

---

## 5. Literature notes

- **Fischer, *Precoding and Signal Shaping for Digital Transmission* (Wiley, 2002)** — Tomlinson–Harashima precoding: a modulo operation bounds transmit amplitude and is reversible at the receiver. Lesson: a mod-q ciphertext has bounded power; GGH has no modulus.
- **Tung & Gündüz, "Deep Joint Source-Channel and Encryption Coding" (DeepJSCEC), arXiv 2208.09245** — LWE encryption at the physical layer for wireless images; channel noise adds to the LWE noise; Bob 25.95 dB PSNR vs Eve 12.29 dB at 10 dB SNR (CIFAR10). ⚠️ They claim the error seed can be public; from their equations, a public seed lets Eve strip the mask. The seed must be **secret** (e.g. derived from ML-KEM).
- **Micciancio (2001), Hermite Normal Form GGH** — smaller key/ciphertext size, but does not bound per-symbol amplitude. GGH is considered broken (Nguyen 1999; Nguyen–Regev 2006).
- **Liu & Sakzad (2024–2025), Kyber as lattice codes / semi-compressed Kyber** — model Kyber decryption as AWGN decoding.
- **Dean & Goldsmith, "Physical-Layer Cryptography Through Massive MIMO", arXiv 1310.1861** — the MIMO channel itself acts as the key; Eve's decoding maps to hard lattice problems.
- **MIMO detection helpers** (few-dB gains, not a fix for the power gap): lattice-reduction-aided detection (Wübben et al., IEEE SPM 2011); integer-forcing receivers (Zhan, Nazer, Erez, Gastpar, IEEE T-IT 2014).

---

## 6. Agreed direction

Keep the supervisor's focus (**sub-band encryption + LDPC over the MIMO uplink**) but change the lattice layer:

1. **Main scheme:** LWE-style (mod-q) encryption per sub-band, with the error seed derived from an **ML-KEM** shared secret. Bounded power, realistic SNR.
2. **Baseline:** current sub-band GGH, reported with the **true SNR**, plus the security-vs-SNR table.
3. **Future work (side note only):** ML-DSA signature or MAC on control messages for authentication/integrity.

---

## 7. Action points

Work inside the project folder. Do not silently change `lattice.py` / `metrics.py`; add options instead and note any change here.

1. **Verify fixes** 2.1 (√10, 3 places), 2.2 (ZF equalizer), 2.4 (seed) are in `PQC_MIMO_Simulations_subband.ipynb`. Add a diagnostic cell: print max |B|, `engine.info`, and a **noiseless test** (Bob's BER must be exactly 0 at very high SNR for every config).
2. **True-SNR option:** add `normalize_power=True` to `Model` and `SubbandModel` (transmit C/√P with a fixed P per key; receiver multiplies by √P). Label plots with true SNR.
3. **Security-vs-SNR sweep** (Shankar Fig. 6 analogue): `LatticeBasedEncryptor(n, rounds=r, max_coeff=c)` for r ∈ {1..5}, c ∈ {1,2,3}; record HR(B), ciphertext power, required true SNR, BER and PSNR.
4. **New `LWESubbandModel`:** per sub-band LWE encryption (start with n=192, q=4093, Gaussian parameter 8.87); map ciphertext (mod q, centred) to a bounded constellation; pre-shared `d` from a secret seed (simulate ML-KEM with `os.urandom`). Decide message packing (1 bit per coefficient at q/2 spacing, or 2 bits at q/4) and report the rate cost. Prototype in numpy first, then in Sionna.
5. **Rerun sweeps** for UMi / UMa / RMa: full-band vs sub-band (32×12, 16×12, and 16×24 over two slots so n=384 stays secure), with/without LDPC, on the true-SNR axis.
6. **Shankar-style outputs** (histogram + stats cells already exist): add a **key-sensitivity** cell (decrypt with a slightly changed private key → garbage image) and an **LDPC code-rate sweep** (1/2, 2/3, 3/4, 5/6).
7. **Frequency selectivity test:** `fft_size = 512` (15.36 MHz) so sub-banding can show a real shape effect.
8. **Image PSNR/SSIM:** average over several channel draws or images; put BER/PSNR in plot titles instead of the fixed word "Clean".

---

## 8. Open questions for the supervisor

- Is switching the main lattice layer from GGH to LWE-style (keeping sub-band + LDPC) acceptable, with GGH as the baseline?
- If the earlier paper's curves were made before the √10 fix, do they need an update?
- Keep static users (allows long time-tiles) or add mobility (limits tile length by coherence time)?

---

## 9. Change log (Claude Code sessions)

- 2026-09-29: saved this handoff as `CLAUDE.md`. Independent check confirmed §2.2 and §4: in an 8×4 Rayleigh test (n=384, current keys) Bob decodes at labelled 0 dB with ZF, but needs labelled 50–60 dB with unit-prior LMMSE; with transmit power normalised, ZF needs ~110–120 dB true SNR (ciphertext power ≈ 121 dB above unit).
- 2026-09-29: **New bug 2.5 found and fixed: Sionna channel ran in single precision.** In Sionna 2.x the UMi/UMa/RMa models default to complex64, so `y` and `h` were complex64 even though the other blocks were complex128. Float32 error (~1e-7 relative) on a ~1e7 GGH ciphertext gives an absolute error of ~4, far above the Babai margin (~0.35). Bob failed **with no noise** (BER 0.0066 for 16×12, 0.12 for 128×3 at 200 dB). This is the most likely cause of the ~1e-3 GGH error floors in §3. Fix: `sionna.phy.config.precision = "double"` in the first cell. Noiseless BER is now exactly 0 for all configs.
- 2026-09-29: notebook `PQC_MIMO_Simulations_subband.ipynb` changes:
  - ZF instead of LMMSE in `Model` and `InterpolationModel` (§2.2). √10 fix (§2.1) and seed (§2.4) were already present.
  - `normalize_power` option in `Model` (and `InterpolationModel`, `HighFidelityImageModel`); `SubbandModel` already had it.
  - New switch cell `TRUE_SNR = True` (after `InterpolationModel`): all GGH sweeps and image cells now use the **true** Eb/N0 axis (80–150 dB). Set `TRUE_SNR = False` for the old labelled axis.
  - New cells: diagnostics (max|B|, HR(B), σ, noiseless BER); GGH security-vs-SNR sweep (AWGN, n=192, rounds 1–5 × max_coeff 1–3); `LWESubbandModel` + sweep (Bob modes, LDPC, Eve) + LWE image transmission at 0–15 dB.
  - Test-image path falls back to `test_image.png` in the project when not on Colab.
- 2026-09-29: new file `lwe.py` (`LWEEncryptor`, Lindner–Peikert n=192, q=4093, s=8.87; Gray-labelled soft LLRs; modes `secret_key`, `exact_mask`, `eve`). `lattice.py` and `metrics.py` are unchanged. New file `LITERATURE.md` (literature search with access status).
- 2026-09-29: measured on the full Sionna chain (UMi, 4 UE × 8 BS antennas, ZF, perfect CSI, double precision):
  - GGH, power normalised: waterfall at 100–110 dB true Eb/N0, error-free from 120 dB (128×3, 32×12, 16×12). Ciphertext power 110–116 dB above unit.
  - LWE 16×12, 1 bit/coef (2 bits/RE): `secret_key` uncoded BER 1e-3 at 10 dB with a small LWE floor (~3e-5); **+ LDPC 1/2: BER 0 from 5 dB**; `exact_mask` uncoded error-free from ~15–20 dB. 2 bits/coef (4 bits/RE) + LDPC 1/2: BER 0 from 5 dB. Eve BER 0.5 at all SNRs.
- 2026-09-29: **Standards-based LWE replaces the Lindner–Peikert prototype** (implemented in the Colab copy `~/Downloads/PQC_MIMO_Simulations_subband.ipynb`, inline, no `lwe.py` needed; original backed up in the session scratchpad).
  - Choice: **ML-KEM (NIST FIPS 203, Module-LWE)** as the main scheme; **FrodoKEM (plain LWE, ISO/IEC 18033-2 Amd 2:2026, BSI/ANSSI-recommended)** as the conservative alternative. GGH stays as the baseline.
  - FIPS 203: K-PKE "shall not be used as a stand-alone scheme". Design: ML-KEM (with FO transform) establishes session key K; per-tile randomness (r, e1, e2) derived from K via SHAKE-256; only `v` is sent; BS rebuilds `u` from K and decrypts `v − sᵀu` (`secret_key`) or subtracts the mask (`keystream`). Confidentiality only.
  - Engine `PQCTileCipher` (ML-KEM-512/768/1024, FrodoKEM-640/976/1344; standard q, noise, bits per coefficient). `PQCSubbandModel`: tile 32×4 = one ML-KEM ciphertext (256 coefficients = 128 REs I/Q).
  - Noise budget (analytic): margin/σ ≈ 18–21 (ML-KEM), 15–20 (FrodoKEM) vs 4.2 (old LP) → no LWE-noise floor. Measured noise std matches theory (ML-KEM-768: 39.1 vs 39.2).
  - Quick MIMO test (UMi, ZF): ML-KEM-768 + LDPC 1/2 BER 0 from 2.5 dB true Eb/N0 (2 bits/RE); FrodoKEM-640 + LDPC 1/2 BER 0 from 5 dB (4 bits/RE, same rate as the GGH 16-QAM setup). Eve BER 0.5.
  - This version was copied into the project as `PQC_MIMO_Simulations_subband.ipynb` (keeping the `subband` Colab badge). The Lindner–Peikert cells were dropped; `lwe.py` is no longer imported by the notebook and is kept only as the earlier prototype.
  - 2026-10-01: new **baby notebook** `GGH_Caveats_baby.ipynb` (presentation stage before the mother notebook): Part A GGH keys; Part B caveats 1–4 (security-vs-SNR sweep, labelled vs true axis in UMi, float32 channel, LMMSE); Part C sub-band Monte Carlo sweep on the true axis ± LDPC, caveat 5 (tile share of the LDPC codeword), Eve check, image at 100–120 dB; Part D scenario checks from Liu et al. 2026 (SVD + Babai keyless scheme on UMi 4×8, Eve ΔH model, BDD/covering-radius check, dimension check); Part E analytic GGH vs ML-KEM/FrodoKEM teaser. Shares `Model`/`SubbandModel` code with the mother notebook.
  - The Downloads notebook also got all GGH fixes (double precision, ZF, `TRUE_SNR`, diagnostics, security-vs-SNR sweep). Its extra histogram cell (sub-band 32×12 + LDPC) now uses the true-SNR axis too.
