# Literature: lattice / LWE encryption at the physical layer and the high-SNR problem

Search date: 2026-09-29. **Access** column: *read* = full text or HTML read; *abstract* = only the abstract / publisher page was reachable (paywalled or blocked, cited from abstract only); *known* = standard reference, not re-read in this search.

## A. Why our GGH scheme needs very high SNR

| Ref | Access | Relevance |
|---|---|---|
| Abdallah et al., "A physical layer security scheme for 6G wireless networks using post-quantum cryptography", *Computer Communications* 218 (2024) 176–187. [link](https://www.sciencedirect.com/science/article/abs/pii/S0140366424000756) | abstract (paywalled, 403) | The GGH-over-OFDM scheme our `lattice.py` follows: QAM vector projected onto the user's public basis, plus a small error vector, before the IFFT. The abstract does not state how the ciphertext power is normalised. **Check in the full text whether their Eb/N0 axis includes the ciphertext power.** |
| Nguyen, "Cryptanalysis of the GGH cryptosystem", CRYPTO 1999; Nguyen & Regev, "Learning a parallelepiped", EUROCRYPT 2006. [link](https://www.researchgate.net/publication/220437582_Cryptanalysis_of_the_GGH_Cryptosystem) | known | GGH is broken in practice. It is only acceptable as a baseline. |
| Micciancio, "Improving lattice based cryptosystems using the Hermite normal form", CaLC 2001 | known | HNF reduces key/ciphertext size, but does **not** bound the amplitude of each symbol, so it does not fix the SNR problem. |
| Kamel et al., "Improving GGH cryptosystem using generalized low density lattices" (2016). [pdf](https://rekaya.wp.imt.fr/files/2017/12/Kamel-GGH-ACOSIS-2016.pdf); "Improving GGH public key scheme using low density lattice codes", [arXiv 1503.03292](https://arxiv.org/pdf/1503.03292) | abstract | LDLC-based GGH variants with better decoding. They still have no modulus, so the same power issue remains. |

## B. The fix: bounded (mod-q) ciphertexts, where channel noise adds to the LWE noise

| Ref | Access | Relevance |
|---|---|---|
| Tung & Gündüz, "Deep Joint Source-Channel and Encryption Coding: Secure Semantic Communications" (DeepJSCEC), ICC 2023, [arXiv 2208.09245](https://arxiv.org/html/2208.09245) | read | **Closest working example of LWE at the physical layer.** Ciphertext in Z_p is mapped to a power-normalised 4096-QAM. Channel noise adds to the LWE noise. Bob gets ~24–28 dB PSNR at **SNR = 10 dB**, over an evaluated range of 2–16 dB. The error terms come from a seed so `d` is not transmitted; they call the seed *public*, but it must be **secret** (see CLAUDE.md §5). |
| Liu & Sakzad, "Lattice codes for CRYSTALS-Kyber", *Des. Codes Cryptogr.* (2025), [arXiv 2308.13981](https://arxiv.org/abs/2308.13981); "Semi-compressed CRYSTALS-Kyber", [arXiv 2407.17684](https://arxiv.org/pdf/2407.17684); "CRYSTALS-Kyber with lattice quantizer", [arXiv 2401.15534](https://arxiv.org/pdf/2401.15534) | abstract (arXiv open) | Kyber decryption treated as decoding over an AWGN channel. The decoding noise is bounded by a sphere, and better lattice codes lower the decryption failure rate by up to 2^85. Gives the theory for "channel noise + LWE noise" in our LWE model and for LDPC as the outer code. |
| Lindner & Peikert, "Better key sizes (and attacks) for LWE-based encryption", CT-RSA 2011 | known | Source of our parameters: n = 192, q = 4093, s = 8.87. It accepts ~1% raw decryption error, fixed by error correction; this matches the floor in our `secret_key` mode. |
| Fischer, *Precoding and Signal Shaping for Digital Transmission*, Wiley 2002 | known (book) | Tomlinson–Harashima precoding: a modulo operation bounds transmit power and is undone at the receiver. Also covers "modulo noise loss" near the constellation edge. Same principle as a mod-q ciphertext. |
| Mirsky, Fedidat, Haddad, "Physical layer encryption using a Vernam cipher" (VPSC), [arXiv 1910.08262](https://arxiv.org/abs/1910.08262) | abstract | Modulo encryption of signal magnitude/phase over a multipath channel, with two noise-mitigation methods ("preemptive-rise", "statistical-floor") for noise near the modulo boundary. A useful precedent for the wrap-around issue. |
| Ling, Luzzi, Belfiore, Stehlé, "Semantically secure lattice codes for the Gaussian wiretap channel", *IEEE T-IT* 60(10) 2014, [arXiv 1210.6673](https://arxiv.org/abs/1210.6673); Campello, Ling, Belfiore, "Semantically secure lattice codes for compound MIMO channels", [arXiv 1903.09954](https://arxiv.org/pdf/1903.09954) | abstract | Information-theoretic lattice secrecy at normal SNRs using mod-Λ channels and the flatness factor. The MIMO version is directly relevant to our uplink. |
| "A survey of lattice-based physical-layer security for wireless systems with p-modular lattice constructions", *Entropy* 28(2) 235 (2026), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12939611/) | abstract (open access) | Recent survey linking lattice cryptography and lattice wiretap coding. Good for the related-work section. |

## B2. Standards for the LWE replacement

| Ref | Access | Relevance |
|---|---|---|
| NIST **FIPS 203**, *Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)*, Aug 2024. [csrc.nist.gov](https://csrc.nist.gov/pubs/fips/203/final) | known / abstract | Main replacement. Module-LWE, n = 256, q = 3329, k = 2/3/4. Its inner scheme K-PKE "shall not be used as a stand-alone scheme" (the FO transform gives the full security), so we use ML-KEM for the session key and K-PKE-style masking inside the session. [Overview](https://www.encryptionconsulting.com/overview-of-fips-203/) |
| **FrodoKEM**, ISO/IEC 18033-2:2006/Amd 2:2026; recommended by BSI (long-term confidentiality) and ANSSI ("conservative option"). [frodokem.org](https://frodokem.org/), [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/frodokem-a-conservative-quantum-safe-cryptographic-algorithm/), [IETF draft](https://datatracker.ietf.org/doc/html/draft-longa-cfrg-frodokem-security-considerations/) | abstract | Plain LWE (no ring structure), n = 640/976/1344, q = 2¹⁵/2¹⁶, 2–4 bits per coefficient. The closest standard to the Lindner–Peikert prototype. |
| NIST FIPS 202 (SHA-3 / SHAKE) | known | XOF used to derive the per-tile randomness from the ML-KEM shared secret. |

## C. MIMO physical-layer cryptography (the channel as the key)

| Ref | Access | Relevance |
|---|---|---|
| Dean & Goldsmith, "Physical-layer cryptography through massive MIMO", ITW 2013 / [arXiv 1310.1861](https://arxiv.org/abs/1310.1861) | abstract | Eve's MIMO decoding with M-PAM is mapped to hard lattice problems (LWE-like). |
| Sakzad & Steinfeld, "Comments on 'Physical-layer cryptography through massive MIMO'", [arXiv 2001.02632](https://arxiv.org/abs/2001.02632) | abstract | An Eve who knows her own channel and Bob's decrypts under the same conditions as Bob. An Eve with many antennas gains an advantage whatever the precoder. In one variant the security conditions stop Bob from decoding uniquely. A warning that "security via SNR/antenna gap" is fragile. |
| Yang Li, "Has MIMO decoding been proved hard from lattice problems?", [arXiv 2609.05013](https://arxiv.org/abs/2609.05013) (Sept 2026) | abstract | Shows the Dean–Goldsmith hardness reduction does not carry over to the **non-modular** MIMO setting. The same "no modulus" issue we hit with GGH, which supports the move to mod-q. |
| "Physical-layer public key encryption through massive MIMO", *ACM AsiaCCS* 2024, [link](https://dl.acm.org/doi/10.1145/3634737.3656284) | abstract (403) | Public-key version of the above, possibly with LWE-based analysis. **Full text needed.** |
| Liu, Cai, Han, Liu, Lu, "Keyless physical-layer cryptography", ISC 2025, LNCS 16186, pp. 24–44, Springer 2026, [doi:10.1007/978-3-032-08124-7_2](https://doi.org/10.1007/978-3-032-08124-7_2) | **read** (Zotero PDF) | The channel is the lattice: SVD precoding s = Vx, and Bob runs Babai on the diagonal Σ (O(N)). Eve's channel estimate is spoiled by a rank-deficient full-duplex DFT pilot design, ΔH ~ CN(0, (P₁−1)/(1+MP₁)) → 1/M. Security is framed as BDD: Bob ‖e₁‖ ≤ λ₁/2; Eve's ΔH·s + e′ beyond the covering radius. USRP B210 2×2 BPSK test (Eve ≈ 50 % from 0–17.5 dB) and MATLAB sims (Eve 128–512 antennas, 4/8/16-QAM, distances). Reproduced in `GGH_Caveats_baby.ipynb` Part D on UMi: Bob works at 0–20 dB, Eve's BER is flat in SNR but stays **below 0.5** with their own ΔH model. Assumptions: passive Eve, full duplex, CSI at Alice. |

## D. Receiver-side helpers (a few dB, not a fix for the power gap)

| Ref | Access | Relevance |
|---|---|---|
| Wübben, Seethaler, Jaldén, Matz, "Lattice reduction: a survey with applications in wireless communications", *IEEE SPM* 28(3) 2011 | known | LR-aided detection for correlated users. |
| Zhan, Nazer, Erez, Gastpar, "Integer-forcing linear receivers", *IEEE T-IT* 2014, [arXiv 1003.5966](https://arxiv.org/abs/1003.5966) | abstract | Decodes integer combinations of streams. Fits **mod-q** codes naturally, so it pairs well with the LWE scheme. |

## E. Experimental / hardware realisations

| Ref | Access | Relevance |
|---|---|---|
| "Hybrid RSA–SHA-256 scheme for enhancing physical-layer security in MIMO-OFDM systems using SDR implementation", *Adv. Technol. Innov.*, [link](https://ojs.imeti.org/index.php/AITI/article/view/15667) | abstract | 2×2 USRP MIMO-OFDM testbed. The scrambling keeps normal BER-vs-SNR with minimal SNR penalty. It works because the encryption keeps symbols on the normal constellation. |
| "Physical-layer security improvement in MIMO OFDM using multilevel chaotic encryption"; "Improved PLS for OFDM using data-based subcarrier scrambling" (ResearchGate) | abstract | Same pattern: encryption that preserves the constellation works at normal SNR. |

No hardware realisation of **GGH-style (non-modular) lattice ciphertexts over the air** was found. All working over-the-air schemes either keep the constellation (scrambling, one-time pad, mod-q) or send ciphertext bits digitally.

## F. Papers named by the user but not located by search

- **Shankar & Mishra (2025), RLWE + LDPC image encryption** (Eb/N0 range 0–12 dB). Not found by web search. Please add the DOI/PDF.
- **Jayasinghe et al.** (SNR range −2.5 to 15 dB). Search found only Jayasinghe et al. (2015) on PLS for relay-assisted MIMO D2D, which may not be the intended paper. Please add the DOI/PDF.

## Take-away for the thesis

1. The high SNR in our results comes from the **non-modular GGH ciphertext** (dynamic range ~10⁸), not from the MIMO channel. The literature (A, C) is consistent with this.
2. Every scheme that works at realistic SNR (B, E) keeps the transmitted signal **bounded**, using mod-q / modulo precoding, a constellation-preserving scramble, or digital transmission of the ciphertext bits.
3. Therefore: LWE (mod-q) per sub-band + LDPC as the main scheme, with GGH as a baseline reported on the true-SNR axis.
