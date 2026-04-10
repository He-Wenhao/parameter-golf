#!/bin/bash
# Hyperparameter sweep for MDLM combined model
# Runs 8 experiments in 2 batches of 4 (one per GPU)
set -e
cd /pscratch/sd/w/whe1/auto-diffusion
source .venv/bin/activate

# Common sweep settings: 500 steps, constant LR, fast eval
COMMON="ITERATIONS=500 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_ITERS=0 WARMUP_STEPS=5 \
TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
ELBO_EVAL_STEPS=32 MAX_EVAL_SEQS=64 NUM_LAYERS=11"

echo "=== BATCH 1: 4 experiments in parallel ==="

# Exp 1: Baseline
CUDA_VISIBLE_DEVICES=0 env $COMMON \
  NOISE_EPS=0.01 TRAIN_SEQ_LEN=1024 LOGIT_SOFTCAP=30 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp1_baseline.log 2>&1 &
P1=$!

# Exp 2: noise_eps=0.1
CUDA_VISIBLE_DEVICES=1 env $COMMON \
  NOISE_EPS=0.1 TRAIN_SEQ_LEN=1024 LOGIT_SOFTCAP=30 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp2_eps01.log 2>&1 &
P2=$!

# Exp 3: seq_len=2048
CUDA_VISIBLE_DEVICES=2 env $COMMON \
  NOISE_EPS=0.01 TRAIN_SEQ_LEN=2048 LOGIT_SOFTCAP=30 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp3_seq2048.log 2>&1 &
P3=$!

# Exp 4: no softcap
CUDA_VISIBLE_DEVICES=3 env $COMMON \
  NOISE_EPS=0.01 TRAIN_SEQ_LEN=1024 LOGIT_SOFTCAP=0 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp4_nocap.log 2>&1 &
P4=$!

echo "Waiting for batch 1..."
wait $P1 $P2 $P3 $P4
echo "Batch 1 done."

echo "=== BATCH 2: 4 experiments in parallel ==="

# Exp 5: cond_dim=128
CUDA_VISIBLE_DEVICES=0 env $COMMON \
  NOISE_EPS=0.01 TRAIN_SEQ_LEN=1024 LOGIT_SOFTCAP=30 COND_DIM=128 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp5_cond128.log 2>&1 &
P5=$!

# Exp 6: dsigma loss
CUDA_VISIBLE_DEVICES=1 env $COMMON \
  NOISE_EPS=0.01 TRAIN_SEQ_LEN=1024 LOGIT_SOFTCAP=30 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=1 \
  python train_mdlm_combined.py > sweep_exp6_dsigma.log 2>&1 &
P6=$!

# Exp 7: combo (eps=0.1 + seq=2048 + nocap)
CUDA_VISIBLE_DEVICES=2 env $COMMON \
  NOISE_EPS=0.1 TRAIN_SEQ_LEN=2048 LOGIT_SOFTCAP=0 COND_DIM=64 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp7_combo.log 2>&1 &
P7=$!

# Exp 8: combo + cond=128
CUDA_VISIBLE_DEVICES=3 env $COMMON \
  NOISE_EPS=0.1 TRAIN_SEQ_LEN=2048 LOGIT_SOFTCAP=0 COND_DIM=128 MLP_MULT=2 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp8_combo_cond128.log 2>&1 &
P8=$!

echo "Waiting for batch 2..."
wait $P5 $P6 $P7 $P8
echo "Batch 2 done."

echo ""
echo "=== SWEEP RESULTS ==="
for f in sweep_exp*.log; do
  name=$(echo $f | sed 's/sweep_//;s/.log//')
  bpb=$(grep "val_bpb" $f | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
  loss=$(grep "val_loss" $f | tail -1 | grep -oP 'val_loss:\K[0-9.]+')
  echo "$name: val_bpb=$bpb val_loss=$loss"
done
