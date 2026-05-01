# Candidate-Aware MLP Baseline

This folder contains a compact baseline package for the mobility-management project.

The MLP baseline uses the same candidate-aware target formulation as the LSTM controller, but removes recurrent memory. This helps separate two effects:

1. the value of restricting target-cell prediction to realistic candidates;
2. the value of sequence modeling with LSTM.

## Model Summary

| Item | Value |
| --- | --- |
| Model type | candidate-aware last-step MLP |
| Candidate pool | K=3 |
| Numeric input dim | 7 |
| Candidate feature dim | 5 |
| Serving cell embedding dim | 32 |
| Hidden size | 256 |
| Dropout | 0.15 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Parameters | 180,311 |

## Offline Test Result

| Metric | Value |
| --- | ---: |
| Test best trigger F1 | 0.6890 |
| Test best trigger threshold | 0.95 |
| Test target accuracy | 0.9089 |
| Test target accuracy on positive rows | 0.8964 |
| Test balanced score | 1.5978 |

## Online Matched 900 s Result

The MLP baseline was also evaluated in matched online ns-3 runs.

| Mode | HO Count | Ping-Pong Rate | Mean Dwell (s) |
| --- | ---: | ---: | ---: |
| MLP baseline | 877.25 | 0.1086 | 28.3938 |

The baseline is useful because it is strong and simple. It confirms that the candidate-aware target formulation is important. The LSTM and hybrid controllers remain useful for studying the broader stability/QoS trade-off in closed-loop simulation.

## Folder Map

```text
checkpoints/        selected small checkpoint for review
csv/                compact result tables
scripts/            MLP training and online evaluation scripts
```

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Full online trace folders are intentionally not included. They are large and not needed for a public code review package.
