# Reproducible GGH MU-MIMO-OFDM experiments

Open `PQC_MIMO_Simulations.ipynb` for the scientific assumptions and ordered experiment driver. The previous notebook is preserved verbatim in `appendix/PQC_MIMO_Simulations_original.ipynb`; existing lattice, metrics, BER and plotting utilities are retained.

## Run

Use the PyTorch Sionna 2.0.1 environment (`requirements-experiments.txt`). From this repository:

```sh
python -m unittest discover -s tests -v
python -m pqc_experiments.validation --output validation.json
```

Then run the notebook. Its validation gate must pass against the current source hash before `run_group` accepts an experiment. Large sweeps require setting `RUN_PAPER_SWEEPS = True`. Results are written to a new directory for each run; existing results are never overwritten. Pilot results have only 2 batches of 2 packets per user/configuration/SNR and must not be treated as statistically conclusive.

## Structure

- `config.py`: immutable configuration and dimension checks.
- `security.py`: fixed key registry, explicit integer QAM domains, two-dimensional tiles, power measurement, Babai/U-inverse recovery, and hard-to-soft baseline.
- `receivers.py`: ordered per-subcarrier MMSE-SIC.
- `model.py`: one configurable Sionna communication model and deterministic channel fixtures.
- `runner.py`: ordered one-factor-family study groups, incremental raw records, latency statistics and threshold summaries.
- `validation.py`: deterministic recovery, coding, receiver, channel and power checks.

## Measurement contracts

BER/BLER operate on information bits; a block is one user frame. Changing coding rate preserves the mother-bit prefix and the fixed resource budget, not payload length. Compatible schemes use identical topology, channel, mother bits, noise, and key. Different scenarios necessarily use different channel models with common seeds. Each raw row stores config, fixture ID and key ID.

Both plain QAM and ciphertext use measured per-user/frame normalization. Pilots retain the same power. One float64 scale per user/frame is assumed available at the receiver. Its ideal transmission time is charged to goodput for all normalized schemes, and recorded separately from ciphertext length overhead (zero). Key exchange and HARQ protocols are not simulated. `gamma` is ciphertext power divided by integer-lattice message power, not normalized QAM power.

GGH errors are counted before and after the U-inverse transformation, without using transmitted information in receiver decisions. Oracle diagnostics run after receiver timing. Fixed BSC-surrogate LLRs follow Babai rounding; they are not true channel likelihoods. LDPC runs the configured fixed iteration count. Ciphertext SIC uses continuous cancellation, conventional QAM SIC uses hard cancellation; estimated SIC noise ignores cancellation errors. No gain is assumed.

CPU key setup latency is one-time. Packet timing excludes topology/channel generation and includes host overhead, with device synchronization and  so why don't you add all the necessary cells discarded warmups. Batch means/std/median/p95 and amortized per-lattice-block encryption/decryption times are reported. Static key tensor storage is distinct from runtime complex matrix copies; neither is peak process/GPU memory. A CUDA run should report its own timings; included CPU times do not predict GPU performance.

Defaults use stationary normalized 3GPP channels, with path loss/shadowing disabled and no hardware or intercarrier impairments. Enable link-budget flags for separate sensitivity studies. n=512, low Hadamard ratios and successful legitimate decryption do not establish cryptographic security.

Threshold estimates require a measured BLER crossing, never extrapolate, and need finer SNR sampling plus many more blocks for a paper. Confidence intervals use an approximate binomial model; correlations can make them optimistic. Zero observed errors is censored evidence, not zero true probability.

## Google Colab

Upload only `PQC_MIMO_Simulations.ipynb` to Colab and Run all. Its first code cell unpacks the embedded source snapshot and installs `requirements-colab.txt`; no GitHub publication is required. Compatible Colab PyTorch builds are retained (>=2.9.1), with Sionna pinned to 2.0.1. The environment cell selects CUDA if available in Colab, otherwise CPU. Validation runs on the selected device. If setup upgrades an already-imported numerical library, follow its restart-session instruction and Run all again.

The final optional cell downloads results and validation as a ZIP. Large simulations still require `RUN_PAPER_SWEEPS = True`. Results are held in temporary Colab storage until downloaded or written to your own mounted Drive. This bootstrap has been tested locally using a simulated Colab surface; an actual hosted Colab/GPU run is not part of the local verification.

After editing experiment modules, regenerate the embedded snapshot with `python scripts/build_colab_bundle.py`. The bundle includes runtime modules, metrics, tests, requirements, documentation and the pilot runner; it excludes historical notebooks, existing results and prior validation reports. Local runs use live repository files and do not install packages automatically.
