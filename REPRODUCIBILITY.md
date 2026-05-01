# Reproducibility Notes

This repository is a compact public extraction of the original experiment workspace. It does not include raw ns-3 traces, SQLite databases, packet captures, build products, or full run folders.

## Environment

Original experiments used:

- Linux
- ns-3.46.1
- Python 3
- PyTorch
- pandas / NumPy
- CSV and SQLite processing

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Stages

1. Build and run an ns-3 scenario from `scenarios/`.
2. Collect RAN telemetry, handover traces, and repository data.
3. Build a candidate-aware dataset with `src/oran_e2_lstm/build_dataset.py`.
4. Train and evaluate the LSTM with `src/oran_e2_lstm/train.py` and `src/oran_e2_lstm/evaluate.py`.
5. Run replay analysis with `src/oran_e2_lstm/replay.py`.
6. Use a persistent Python worker for online inference:

```bash
python src/oran_e2_lstm/persistent_inference_worker.py --help
```

## Controls Used in the Project

- Split by simulation runs, not random rows.
- Fit normalization only on the train split.
- Select model and thresholds on validation data.
- Use the test split once for final supervised evaluation.
- Compare online controllers against A3 under matched seeds/runs.
- Report both mobility-stability and QoS metrics.

## Public Package Limitations

The included files are enough to review the method and code structure. Exact numeric reproduction requires regenerating the raw ns-3 traces and rebuilding derived datasets.

The following are intentionally excluded:

- raw `*.tr` traces;
- SQLite `*.db` / `*.sqlite` files;
- full ns-3 source tree and build products;
- large result folders;
- private publication documents and final paper PDFs.

