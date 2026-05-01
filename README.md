# Predictive Mobility Management in LTE/O-RAN with ns-3 and LSTM

## О проекте

Публичный research package по проекту интеллектуального управления мобильностью в LTE / Open RAN-like сценарии.

Проект отвечает на прикладной RAN-вопрос: может ли ML-контроллер использовать компактную радиотелеметрию для поддержки handover-решений и при этом балансировать стабильность мобильности и QoS.

> Статус статьи: связанная работа **"Candidate-aware Long Short Term Memory Handover Control in a Simulated LTE Scenario with Open RAN"** принята на IEEE EDM 2026, UID 6217. Финальный PDF статьи, подписанные документы, письма конференции и скриншоты страниц статьи намеренно не включены в репозиторий.

## Почему это релевантно YADRO

Проект находится на стыке разработки и моделирования телеком-систем:

- C++-сценарии `ns-3` для multi-cell LTE/O-RAN-like симуляций;
- сбор RAN-телеметрии: `RSRP`, `RSRQ`, `SINR`, `BLER/TBLER`, throughput, delay, handover events;
- Python/PyTorch pipeline для dataset preparation, training, replay и online inference;
- candidate-aware LSTM для handover trigger и target-cell prediction;
- MLP baseline для проверки вклада candidate-aware постановки;
- online closed-loop проверка в симуляторе и сравнение с A3 baseline.

## Краткая схема

```mermaid
flowchart LR
    A[ns-3 LTE сценарий] --> B[RAN telemetry и HO traces]
    B --> C[CSV / SQLite processing]
    C --> D[Candidate-aware dataset]
    D --> E[LSTM training]
    D --> F[MLP baseline]
    E --> G[Offline test и replay]
    F --> G
    G --> H[Online closed-loop ns-3 validation]
    H --> I[Mobility и QoS metrics]
```

## Основные результаты

Offline test для candidate-aware LSTM K=3:

| Метрика | Значение |
| --- | ---: |
| Trigger F1 | 0.6888 |
| Candidate target accuracy | 0.8967 |
| Candidate macro-F1 | 0.8973 |
| Candidate hit rate | 0.9862 |

Final matched 900 s online summary over runs 1, 3, 5, and 6:

| Режим | HO Count | Ping-Pong Rate | Mean Dwell (s) | DL Throughput (Mbps) |
| --- | ---: | ---: | ---: | ---: |
| A3 | 2329.50 | 0.2673 | 10.9621 | 33.8246 |
| LSTM-only | 1477.50 | 0.1863 | 17.1526 | 31.4682 |
| LSTM+A3 hybrid | 2441.75 | 0.2539 | 10.4093 | 34.0699 |

Интерпретация:

- `LSTM-only` снижает число handover и ping-pong, увеличивает среднее время пребывания в соте.
- `LSTM+A3 hybrid` лучше сохраняет QoS/throughput и ближе к A3 по сервисным метрикам.
- Главный вывод: меньше handover не всегда означает лучшее качество сервиса, поэтому модель нужно оценивать по нескольким телеком-метрикам одновременно.

## Что смотреть в первую очередь

1. [`scenarios/lte-oran-helper-lstm-hex7.cc`](scenarios/lte-oran-helper-lstm-hex7.cc) - интеграция online-контроллера в ns-3.
2. [`src/oran_e2_lstm/model.py`](src/oran_e2_lstm/model.py) - candidate-aware LSTM.
3. [`src/oran_e2_lstm/persistent_inference_worker.py`](src/oran_e2_lstm/persistent_inference_worker.py) - persistent Python inference worker.
4. [`src/oran_e2_lstm/replay.py`](src/oran_e2_lstm/replay.py) - replay-анализ политик.
5. [`baselines/mlp/README.md`](baselines/mlp/README.md) - MLP baseline.
6. [`docs/YADRO_REVIEW_GUIDE_RU.md`](docs/YADRO_REVIEW_GUIDE_RU.md) - краткий гид на русском.

## Что не включено

Репозиторий намеренно сделан компактным. В него не включены:

- финальный PDF статьи;
- скриншоты или фрагменты страниц статьи;
- экспертные заключения, договоры, подписи и печати;
- письма конференции и переписка с рецензентами;
- raw traces, SQLite databases, packet captures и полный `ns-3` tree.

---

## English Overview

Public research package for an AI-assisted mobility management project in a simulated LTE / Open RAN-like scenario.

The project asks a practical RAN question: can a learned controller use compact radio telemetry to support handover decisions while balancing mobility stability and QoS?

> Paper status: the related paper, **"Candidate-aware Long Short Term Memory Handover Control in a Simulated LTE Scenario with Open RAN"**, has been accepted to IEEE EDM 2026, UID 6217. The final paper PDF, signed documents, conference e-mails, and article screenshots are intentionally not included in this repository.

## Why This Repository Exists

This repository is a compact, public-ready extraction of the implementation and experimental artifacts:

- ns-3 C++ scenarios for multi-cell LTE/O-RAN-like mobility experiments;
- Python dataset, model, replay, and online inference code;
- candidate-aware LSTM controller for handover trigger and target-cell prediction;
- candidate-aware MLP baseline for method comparison;
- compact CSV/Markdown result summaries;
- reproducibility notes and publication-safety notes.

Large raw traces, SQLite databases, packet captures, full ns-3 sources, private documents, and final conference PDFs are excluded.

## Project Pipeline

```mermaid
flowchart LR
    A[ns-3 multi-cell LTE scenario] --> B[RAN telemetry and HO traces]
    B --> C[CSV / SQLite processing]
    C --> D[Candidate-aware dataset]
    D --> E[LSTM training]
    D --> F[MLP baseline]
    E --> G[Offline test and replay]
    F --> G
    G --> H[Online closed-loop ns-3 validation]
    H --> I[Mobility and QoS metrics]
```

## Online Decision Path

```mermaid
flowchart LR
    A[UE / serving cell telemetry] --> B[C++ ns-3 controller loop]
    B --> C[Persistent Python inference worker]
    C --> D[LSTM or MLP checkpoint]
    D --> E[Trigger probability + target ranking]
    E --> F[Safety filters: confidence, cooldown, anti-ping-pong]
    F --> G{Issue handover?}
    G -->|yes| H[HO command]
    G -->|no| I[Keep serving cell]
    H --> J[Metrics and logs]
    I --> J
```

## Repository Map

```text
scenarios/              ns-3 C++ scenario files and run script
src/oran_e2_lstm/       Python dataset, model, training, replay, and inference code
baselines/mlp/          MLP baseline scripts, compact CSV results, and reports
results/                final compact CSV/Markdown summaries
docs/                   project notes for reviewers
```

## Main Technical Idea

The controller receives compact E2-friendly observations and predicts:

1. whether a handover should be triggered soon;
2. which target cell should be selected from a realistic candidate set.

The key modeling change is the **candidate-aware target formulation**. Instead of predicting one class from all cells, the model scores a small realistic candidate set. This better matches how handover control is usually reasoned about in a radio network.

## Selected Offline Results

Candidate-aware LSTM K=3 on the held-out test split:

| Metric | Value |
| --- | ---: |
| Trigger F1 | 0.6888 |
| Candidate target accuracy | 0.8967 |
| Candidate macro-F1 | 0.8973 |
| Candidate hit rate | 0.9862 |

Candidate-aware MLP baseline:

| Metric | Value |
| --- | ---: |
| Parameters | 180,311 |
| Test trigger F1 | 0.6890 |
| Test target accuracy | 0.9089 |

## Selected Online Results

Final matched 900 s online summary over runs 1, 3, 5, and 6:

| Mode | HO Count | Ping-Pong Rate | Mean Dwell (s) | DL Throughput (Mbps) |
| --- | ---: | ---: | ---: | ---: |
| A3 | 2329.50 | 0.2673 | 10.9621 | 33.8246 |
| LSTM-only | 1477.50 | 0.1863 | 17.1526 | 31.4682 |
| LSTM+A3 hybrid | 2441.75 | 0.2539 | 10.4093 | 34.0699 |

Interpretation:

- `LSTM-only` is the stability-first mode: fewer handovers, lower ping-pong, longer dwell time.
- `LSTM+A3 hybrid` is the QoS-preserving predictive mode: closer to A3 service metrics.
- A3 remains the non-predictive reference baseline.

## Quick Start

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected ns-3 workflow:

```bash
export NS3_ROOT=/path/to/ns-allinone-3.46.1/ns-3.46.1
cp scenarios/*.cc "$NS3_ROOT/scratch/"
cp scenarios/run_v7_lstm_sequence.sh "$NS3_ROOT/"
```

Then configure these environment variables for your local setup:

```bash
export PYTHON_BIN=/path/to/python
export INFERENCE_SCRIPT=/path/to/persistent_inference_worker.py
export CHECKPOINT_PATH=/path/to/best_model.pt
```

Run a scenario from the ns-3 root after adapting paths and building ns-3.

## What to Review First

For a quick technical review:

1. `scenarios/lte-oran-helper-lstm-hex7.cc` - ns-3 online controller integration.
2. `src/oran_e2_lstm/model.py` - candidate-aware LSTM model.
3. `src/oran_e2_lstm/persistent_inference_worker.py` - low-overhead inference path.
4. `src/oran_e2_lstm/replay.py` - replay policy evaluation.
5. `baselines/mlp/README.md` - MLP baseline protocol and results.
6. `results/final_run_numbers/final_900s_runs1_3_5_6_summary.md` - compact online summary.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The public package is intentionally compact. It is enough to review the method and code structure, but exact numeric reproduction requires regenerating raw ns-3 traces.

## Publication and Privacy Notes

See [docs/PUBLICATION_AND_PRIVACY.md](docs/PUBLICATION_AND_PRIVACY.md).

This repository should not include:

- final article PDF;
- screenshots/crops of article pages;
- signed expert conclusions;
- contracts, personal documents, stamps, or signatures;
- conference e-mails or private review correspondence;
- raw traces and databases.

## License

License is intentionally not finalized in this package. The Python code can likely use a permissive license if fully authored by the project owner. The C++ scenario files depend on ns-3, so license compatibility should be checked before publishing.
