#!/usr/bin/env bash

set -euo pipefail

ROOT="${NS3_ROOT:-/path/to/ns-allinone-3.46.1/ns-3.46.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_SCRIPT="${INFERENCE_SCRIPT:-/path/to/oran-lstm-handover/src/oran_e2_lstm/persistent_inference_worker.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoint.pt}"
LSTM_ONLY_BIN="${LSTM_ONLY_BIN:-$ROOT/build/optimized/scratch/ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized}"
HYBRID_BIN="${HYBRID_BIN:-$ROOT/build/optimized/scratch/ns3.46.1-lte-oran-helper-lstm-hex7-optimized}"

SIM_TIME="${SIM_TIME:-900}"
SEED="${SEED:-12345}"
RUN_ID="${RUN_ID:-1}"
SEQ_LEN="${SEQ_LEN:-32}"
TAU="${TAU:-0.20}"
GAMMA="${GAMMA:-0.50}"
BETA="${BETA:-0.02}"
MIN_CONF="${MIN_CONF:-0.20}"
COOLDOWN="${COOLDOWN:-2.0}"
DIST_TOPK="${DIST_TOPK:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

LSTM_ONLY_ROOT="$ROOT/results_night/sim_v7_e2_lstm_only_seq${SEQ_LEN}_run${RUN_ID}_${SIM_TIME}s_${STAMP}"
HYBRID_ROOT="$ROOT/results_night/sim_v7_e2_a3_lstm_seq${SEQ_LEN}_run${RUN_ID}_${SIM_TIME}s_${STAMP}"

if [ ! -x "$LSTM_ONLY_BIN" ]; then
  echo "LSTM_ONLY_BIN is not executable: $LSTM_ONLY_BIN" >&2
  echo "Build the ns-3 scratch scenario first or set LSTM_ONLY_BIN explicitly." >&2
  exit 1
fi

if [ ! -x "$HYBRID_BIN" ]; then
  echo "HYBRID_BIN is not executable: $HYBRID_BIN" >&2
  echo "Build the ns-3 scratch scenario first or set HYBRID_BIN explicitly." >&2
  exit 1
fi

if [ ! -f "$RUNTIME_SCRIPT" ]; then
  echo "INFERENCE_SCRIPT does not exist: $RUNTIME_SCRIPT" >&2
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "v7 sequence start"
echo "seed=$SEED run=$RUN_ID sim_time=$SIM_TIME seq_len=$SEQ_LEN tau=$TAU gamma=$GAMMA beta=$BETA cooldown=$COOLDOWN dist_topk=$DIST_TOPK"
echo "lstm_only_output=$LSTM_ONLY_ROOT"
echo "hybrid_output=$HYBRID_ROOT"

"$LSTM_ONLY_BIN" \
  --seed="$SEED" \
  --run="$RUN_ID" \
  --sim-time="$SIM_TIME" \
  --outputRoot="$LSTM_ONLY_ROOT" \
  --enableLstmController=1 \
  --useLteHandover=0 \
  --lstmDecisionIntervalSec=0.1 \
  --lstmSeqLen="$SEQ_LEN" \
  --lstmTriggerThreshold="$TAU" \
  --lstmTargetThreshold="$GAMMA" \
  --lstmUtilityThreshold="$BETA" \
  --lstmTargetDistanceTopK="$DIST_TOPK" \
  --lstmMinConfidence="$MIN_CONF" \
  --lstmCooldownSec="$COOLDOWN" \
  --lstmPreferNonServingTarget=1 \
  --lstmPythonPath="$PYTHON_BIN" \
  --lstmInferenceScript="$RUNTIME_SCRIPT" \
  --lstmCheckpointPath="$CHECKPOINT_PATH"

echo "lstm_only_completed"

"$HYBRID_BIN" \
  --seed="$SEED" \
  --run="$RUN_ID" \
  --sim-time="$SIM_TIME" \
  --outputRoot="$HYBRID_ROOT" \
  --enableLstmController=1 \
  --useLteHandover=1 \
  --lstmDecisionIntervalSec=0.1 \
  --lstmSeqLen="$SEQ_LEN" \
  --lstmTriggerThreshold="$TAU" \
  --lstmTargetThreshold="$GAMMA" \
  --lstmUtilityThreshold="$BETA" \
  --lstmTargetDistanceTopK="$DIST_TOPK" \
  --lstmMinConfidence="$MIN_CONF" \
  --lstmCooldownSec="$COOLDOWN" \
  --lstmPreferNonServingTarget=1 \
  --lstmPythonPath="$PYTHON_BIN" \
  --lstmInferenceScript="$RUNTIME_SCRIPT" \
  --lstmCheckpointPath="$CHECKPOINT_PATH"

echo "hybrid_completed"
