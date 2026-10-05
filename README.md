# Wireless Simulation with Lattice-Based Encryption

1. **Setup**:
Create a virtual environment, activate it, and install the dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Run**:
Launch the Jupyter notebook and run all cells:
```bash
jupyter notebook simulation.ipynb
```

## All PQC experiments in Colab

Use [PQC_Colab_All_Experiments.ipynb](PQC_Colab_All_Experiments.ipynb) as the companion execution/reporting notebook. The original [PQC_MIMO_Simulations.ipynb](PQC_MIMO_Simulations.ipynb) remains the detailed model and methodology notebook. Both import the same `pqc_experiments` implementation.

1. Push the companion notebook, `pqc_experiments/`, `metrics.py`, and `requirements-colab.txt` to the `rev1` branch (or change `GIT_REF` in the companion notebook). Push `results/pilot/` too if you want to browse those historical results in Colab. Do not commit Python caches or local virtual environments.
2. Upload the companion notebook to Colab, or open it from GitHub after pushing. Select a GPU runtime if desired.
3. Run all. The default runs validation and all ten experiment families at pilot scale, showing tables and figures after each. K and M sensitivity use separate subgroups, producing eleven group folders.
4. Inspect the completion table and download the ZIP from the final cell, or enable Drive storage before running. Completed matching groups are reused on cell reruns; incomplete attempts are preserved and retried separately.

Set `MODE = "view_saved"` to visualize `results/pilot/` without new simulations. Set `SAVED_RESULTS_ROOT` to another saved run or an extracted Colab download if needed. Set `PROFILE = "extended"` only after reviewing the workload preview; a finer grid and more independent trials are still needed for paper-level thresholds.

The notebook does not push code or results to GitHub. It records the checked-out commit and source hash. Existing Colab checkouts are reused without pulling; start a fresh runtime when you want newly pushed source. A dependency upgrade may require a session restart followed by Run all.
