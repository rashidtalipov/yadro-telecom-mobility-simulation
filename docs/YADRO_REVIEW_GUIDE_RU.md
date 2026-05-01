# Краткий гид для технического просмотра

Этот репозиторий показывает проект на стыке телекоммуникаций, моделирования, C++/Python и ML.

## Что это за проект

Задача проекта - проверить, может ли learned controller помогать принимать handover-решения в LTE/O-RAN-like сценарии на основе RAN-телеметрии.

Ключевые данные:

- `RSRP`, `RSRQ`, `SINR`;
- `BLER/TBLER`;
- `throughput`, `delay`;
- serving/target cell;
- handover events;
- mobility traces.

## Почему это релевантно команде разработки и моделирования телеком-систем

В проекте есть:

- C++-интеграция с `ns-3`;
- симуляционный контур для multi-cell LTE сценария;
- работа с handover, A3 baseline и target-cell selection;
- Python/PyTorch pipeline для модели;
- сравнение offline/replay/online результатов;
- анализ компромисса между mobility stability и QoS.

## Что смотреть в первую очередь

1. `README.md` - общий обзор, диаграммы, основные результаты.
2. `scenarios/lte-oran-helper-lstm-hex7.cc` - online-контроллер в ns-3.
3. `src/oran_e2_lstm/model.py` - candidate-aware LSTM.
4. `src/oran_e2_lstm/persistent_inference_worker.py` - persistent inference worker.
5. `baselines/mlp/README.md` - MLP baseline и протокол сравнения.
6. `results/final_run_numbers/final_900s_runs1_3_5_6_summary.md` - компактная таблица финальных online-результатов.

## Главное инженерное наблюдение

Меньше handover не всегда означает лучшее качество сервиса. В online runs `LSTM-only` улучшал stability metrics, но мог снижать QoS, а `LSTM+A3 hybrid` лучше сохранял throughput и delivery metrics. Поэтому проект оценивает не одну ML-accuracy, а несколько телеком-метрик одновременно.

