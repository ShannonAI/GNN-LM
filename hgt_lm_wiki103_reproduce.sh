DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
QUANTIZER=$DATA_BIN/quantizer

n=1024
k=32
c=1
lr=2e-5
PRETRAINED="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best_quantize.pt"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n${n}_k${k}_adam_lr$lr"

mkdir -p $MODEL_DIR
LOG=$MODEL_DIR/log.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # todo change lr; gcn dropout

# Stage 1: train with gcn-k=32
fairseq-train --task language_modeling \
    $DATA_BIN \
    --pretrained_part $PRETRAINED \
    --save-dir $MODEL_DIR \
    --arch transformer_lm_wiki103 --quantizer_path $QUANTIZER --freeze --use-precompute-feat \
    --graph --neighbor-context $c --graph_layer 3 --decoder_gcn_dim 1024 --gcn-k $k --invalid-neighbor-context 2560 \
    --optimizer adam --adam-betas '(0.9, 0.98)'  \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --lr $lr --min-lr 1e-09 \
    --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens $n --update-freq 3 --tokens-per-sample $n --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --update-freq 3 \
    --num-workers 2 --save-interval-updates 1000 --log-interval 100 \
    >$LOG 2>&1 & tail -f $LOG


# Stage 2: train with gcn-k=128, init with best checkpoint from stage 1.
n=256
k=128
c=1
lr=2e-5
PRETRAINED="/userhome/yuxian/train_logs/lm/wiki-103/0709_gcn_bidirect_noleak_adam_lr1e-5/checkpoint_best.pt"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n${n}_k${k}_adam_lr$lr"

fairseq-train --task language_modeling \
    $DATA_BIN \
    --pretrained_part $PRETRAINED \
    --save-dir $MODEL_DIR \
    --arch transformer_lm_wiki103 --quantizer_path $QUANTIZER --freeze --use-precompute-feat \
    --graph --neighbor-context $c --graph_layer 3 --decoder_gcn_dim 1024 --gcn-k $k --invalid-neighbor-context 2560 \
    --optimizer adam --adam-betas '(0.9, 0.98)'  \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --lr $lr --min-lr 1e-09 \
    --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens $n --update-freq 3 --tokens-per-sample $n --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --update-freq 3 \
    --num-workers 2 --save-interval-updates 1000 --log-interval 100 \
    >$LOG 2>&1 & tail -f $LOG



# evaluate
DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n256_k128_adam_lr2e-5"
MODEL_NAME="checkpoint_best.pt"

QUANTIZER=$DATA_BIN/quantizer

subset="test"
c=1
k=128
n=1024
alpha=0.0
LOG=$MODEL_DIR/ppl_${subset}.txt.k${k}.n${n}.alpha${alpha}  # todo 为什么模型没法overwrite tokens-per-sample??
CUDA_VISIBLE_DEVICES=2 fairseq-eval-lm $DATA_BIN \
    --path $MODEL_DIR/$MODEL_NAME \
    --graph --neighbor-context $c  --gcn-k $k --use-precompute-feat \
    --sample-break-mode none --max-tokens $n --tokens-per-sample $n \
     --softmax-batch 3072 \
    --model-overrides "{'orig_prob_ratio': $alpha, 'quantizer_path': '${QUANTIZER}', 'max_target_positions': $n }" \
    --gen-subset $subset  \
    > $LOG 2>&1 & tail -f $LOG


# extract gcn feature
DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0709_gcn_bidirect_noleak"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n256_k128_adam_lr2e-5"

MODEL_NAME="checkpoint_best.pt"
MODEL_NAME="checkpoint_1_8000.pt"

QUANTIZER=$DATA_BIN/quantizer

for subset in "test" "valid" "train";do
c=1
k=128
n=256
alpha=0.0
CUDA_VISIBLE_DEVICES=5 fairseq-eval-lm $DATA_BIN \
    --path $MODEL_DIR/$MODEL_NAME \
    --graph --neighbor-context $c  --gcn-k $k --use-precompute-feat \
    --sample-break-mode none --max-tokens $n --tokens-per-sample $n  --softmax-batch 3072 \
    --model-overrides "{'orig_prob_ratio': $alpha, 'quantizer_path': '${QUANTIZER}', 'max_target_positions': $n }" \
    --gen-subset $subset \
    --dstore-mmap $DATA_BIN \
    --save-knnlm-dstore --dstore-fp16 --knn-keytype "gcn_feat"
done



# eval with knn
DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n256_k128_adam_lr2e-5"
DSTORE=$DATA_BIN/train_dstore-gcn_feat
INDEX=$DSTORE/faiss_store.cosine
QUANTIZER=$DATA_BIN/quantizer
subset="test"
c=1
k=128
n=256
alpha=0.0
LOG=$MODEL_DIR/ppl_${subset}_knn.txt
CUDA_VISIBLE_DEVICES=5 fairseq-eval-lm $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --graph --neighbor-context $c  --gcn-k $k --use-precompute-feat \
    --sample-break-mode none --max-tokens $n --tokens-per-sample $n  --softmax-batch 3072 \
    --model-overrides "{'orig_prob_ratio': $alpha, 'quantizer_path': '${QUANTIZER}', 'max_target_positions': $n }" \
    --gen-subset $subset \
    --knnlm --knn-keytype "gcn_feat" \
    --k 1024 --lmbda 0.15 --dstore-dir $DSTORE --index-file $INDEX \
    --temperature 0.01  \
    > $LOG 2>&1 & tail -f $LOG