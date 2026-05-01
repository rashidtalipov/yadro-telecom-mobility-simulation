# Final 900 s Combined Results for Runs 1, 3, 5, and 6

This file consolidates the final long-run numbers from the expanded matched evaluation.

## Aggregate 4-Run Summary

| Mode | Runs | HO Count | PP Rate | Rapid Returns | Mean Dwell (s) | DL Delivery | UL Delivery | DL Throughput (Mbps) | UL Throughput (Mbps) | DL Delay (ms) | UL Delay (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | 1, 3, 5, 6 | 2329.50 | 0.2673 | 630.75 | 10.9621 | 0.7401 | 0.5222 | 33.8246 | 8.2520 | 422.4 | 2782.2 |
| LSTM-only | 1, 3, 5, 6 | 1477.50 | 0.1863 | 274.50 | 17.1526 | 0.6884 | 0.4927 | 31.4682 | 7.7875 | 722.9 | 4341.0 |
| LSTM+A3 hybrid | 1, 3, 5, 6 | 2441.75 | 0.2539 | 623.00 | 10.4093 | 0.7456 | 0.5243 | 34.0699 | 8.2844 | 401.2 | 2880.7 |

## Per-Run Values

| Run | Mode | HO Count | PP Rate | Rapid Returns | Mean Dwell (s) | DL Delivery | UL Delivery | DL Throughput (Mbps) | UL Throughput (Mbps) | DL Delay (ms) | UL Delay (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A3 | 2023 | 0.2595 | 534 | 12.3891 | 0.7616 | 0.5107 | 35.2061 | 8.1015 | 406.9 | 3145.0 |
| 1 | LSTM-only | 1268 | 0.1893 | 241 | 19.5836 | 0.7361 | 0.4714 | 34.0336 | 7.4789 | 513.7 | 3453.1 |
| 1 | LSTM+A3 hybrid | 2156 | 0.2778 | 608 | 11.5662 | 0.7709 | 0.5070 | 35.6237 | 8.0415 | 345.2 | 2766.5 |
| 3 | A3 | 2335 | 0.2737 | 645 | 10.9427 | 0.7155 | 0.4772 | 32.7459 | 7.5433 | 415.7 | 3882.1 |
| 3 | LSTM-only | 1279 | 0.1955 | 250 | 19.1792 | 0.6124 | 0.4214 | 28.0244 | 6.6618 | 837.6 | 5457.1 |
| 3 | LSTM+A3 hybrid | 2437 | 0.2524 | 621 | 10.5333 | 0.7180 | 0.4734 | 32.8481 | 7.4812 | 396.9 | 4380.8 |
| 5 | A3 | 2292 | 0.2731 | 630 | 10.6850 | 0.7671 | 0.5797 | 34.8305 | 9.1668 | 398.9 | 2174.7 |
| 5 | LSTM-only | 1663 | 0.1834 | 305 | 14.5681 | 0.7270 | 0.5791 | 33.0214 | 9.1586 | 623.8 | 2850.7 |
| 5 | LSTM+A3 hybrid | 2432 | 0.2438 | 598 | 9.8885 | 0.7716 | 0.5872 | 35.0340 | 9.2828 | 390.2 | 2477.3 |
| 6 | A3 | 2668 | 0.2627 | 714 | 9.8317 | 0.7162 | 0.5211 | 32.5158 | 8.1964 | 468.2 | 1926.8 |
| 6 | LSTM-only | 1700 | 0.1771 | 302 | 15.2794 | 0.6781 | 0.4990 | 30.7933 | 7.8506 | 916.5 | 5603.1 |
| 6 | LSTM+A3 hybrid | 2742 | 0.2414 | 665 | 9.6494 | 0.7218 | 0.5297 | 32.7740 | 8.3322 | 472.3 | 1898.1 |

## Practical Note

- `LSTM-only` remains the strongest stability-first mode in the expanded 4-run long evaluation.
- `LSTM+A3 hybrid` becomes the strongest QoS-preserving predictive mode in the expanded 4-run long evaluation.
- `worker_failed_requests = 0` in all predictive runs in this merged set.
