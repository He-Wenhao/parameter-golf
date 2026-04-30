#!/bin/bash
# Follow-up sweep: eps=0.1 confirmed best, now test architecture variations
set -e
cd /pscratch/sd/w/whe1/auto-diffusion
source .venv/bin/activate

COMMON="ITERATIONS=500 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_ITERS=0 WARMUP_STEPS=5 \
TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
ELBO_EVAL_STEPS=32 MAX_EVAL_SEQS=64 NOISE_EPS=0.1 LOGIT_SOFTCAP=30 COND_DIM=64"

echo "=== SWEEP 2: 4 experiments in parallel (all with eps=0.1) ==="

# Exp 9: eps=0.1 + seq=2048
CUDA_VISIBLE_DEVICES=0 env $COMMON \
  NUM_LAYERS=11 MLP_MULT=2 TRAIN_SEQ_LEN=2048 TIE_EMBEDDINGS=1 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp9_eps01_seq2048.log 2>&1 &
P1=$!

# Exp 10: eps=0.1 + no weight tying
CUDA_VISIBLE_DEVICES=1 env $COMMON \
  NUM_LAYERS=11 MLP_MULT=2 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=0 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp10_eps01_notie.log 2>&1 &
P2=$!

# Exp 11: eps=0.1 + 9L + 3x MLP (wider model, fewer layers)
CUDA_VISIBLE_DEVICES=2 env $COMMON \
  NUM_LAYERS=9 MLP_MULT=3 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp11_eps01_9L_3x.log 2>&1 &
P3=$!

# Exp 12: eps=0.1 + 9L + 3x MLP + seq=2048
CUDA_VISIBLE_DEVICES=3 env $COMMON \
  NUM_LAYERS=9 MLP_MULT=3 TRAIN_SEQ_LEN=2048 TIE_EMBEDDINGS=1 USE_DSIGMA_LOSS=0 \
  python train_mdlm_combined.py > sweep_exp12_eps01_9L_3x_seq2048.log 2>&1 &
P4=$!

echo "Waiting..."
wait $P1 $P2 $P3 $P4

echo ""
echo "=== SWEEP 2 RESULTS ==="
for f in sweep_exp{9,10,11,12}*.log; do
  name=$(echo $f | sed 's/sweep_//;s/.log//')
  bpb=$(grep "val_bpb" $f | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
  params=$(grep -oP '[0-9,]+ params' $f | head -1)
  artifact=$(grep "artifact:" $f | grep -oP 'artifact:\K[0-9]+ bytes')
  echo "$name: val_bpb=$bpb params=$params artifact=$artifact"
done
