# Reproducibility Notes

This file records the exact workflow and commands used for the validation-
selected MLP baseline.

## Environment

Project root:

```bash
cd <NS3_ROOT>
```

Python environment:

```text
results_night/.venv/bin/python
```

Important packages observed during the run:

```text
torch 2.10.0+cu128
pandas 2.3.3
scikit-learn 1.8.0
numpy 2.3.5
```

ns-3 binary:

```text
build/scratch/ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized
```

## Offline sweep command

```bash
results_night/.venv/bin/python results_night/oran_e2_lstm/mlp_sweep.py \
  --dataset-dir results_night/oran_e2_lstm/processed_candidate_e2_100ms_full \
  --output-dir results_night/oran_e2_lstm/mlp_sweeps/paper_val_selected_20260419_010927 \
  --variant-set paper \
  --candidate-top-k 3 \
  --device auto \
  --seed 12345 \
  --batch-size 8192 \
  --epochs 6 \
  --patience 3 \
  --train-samples-per-epoch 1200000 \
  --max-train-rows 1500000 \
  --max-val-rows 500000 \
  --max-test-rows -1 \
  --trigger-threshold 0.70 \
  --target-loss-weight 1.5 \
  --global-loss-weight 0.2 \
  --selection-metric validation_balanced_score
```

Important methodological point: `--evaluate-test-for-all` was not used. The
script loaded train/validation first, trained all variants, selected the winner
by validation score, then loaded/evaluated the test set only for the selected
variant.

## Online matched run command

The matched online batch was launched with:

```bash
results_night/.venv/bin/python results_night/oran_e2_lstm/run_mlp_matched_900s.py \
  --output-root results_night/oran_e2_lstm/online_runs/mlp_val_selected_matched_900s_20260419_014610 \
  --runs 1 3 4 5 \
  --sim-time 900 \
  --poll-interval-s 300 \
  --trigger-threshold 0.85 \
  --target-threshold 0.70
```

The runner executed the four ns-3 runs sequentially and updated the report
after each run. All four runs completed with exit code `0`.

## Proof that MLP was used online

The ns-3 controller path still uses historical `lstm*` option names, but the
actual worker and checkpoint were MLP:

```text
--lstmInferenceScript=.../results_night/oran_e2_lstm/persistent_mlp_worker.py
--lstmCheckpointPath=.../mlp_sweeps/paper_val_selected_20260419_010927/wide_h256_d15_lr5e4/best_model.pt
```

The checkpoint itself contains:

```text
model_type: candidate_aware_last_step_mlp
```

The run-info files for `run=1,3,4,5` all record `persistent_mlp_worker.py` and
the selected MLP checkpoint.

## Source files copied into this folder

```text
scripts/mlp_sweep.py
scripts/run_mlp_matched_900s.py
scripts/persistent_mlp_worker.py
```

These are copies for documentation. The original files remain in:

```text
ns-allinone-3.46.1/ns-3.46.1/results_night/oran_e2_lstm/
```

## Full online traces

The full online trace folder is not duplicated. Use the symlink:

```text
MLP/links/full_online_runs_mlp_val_selected_matched_900s
```

or the original path:

```text
ns-allinone-3.46.1/ns-3.46.1/results_night/oran_e2_lstm/online_runs/mlp_val_selected_matched_900s_20260419_014610
```

