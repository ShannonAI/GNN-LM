#DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
#MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline"
DATA_BIN="/data/yuxian/datasets/wikitext-103/data-bin"
MODEL_DIR="/data/nfsdata2/nlp_application/models/language_models/debug"
mkdir -p $MODEL_DIR
LOG=$MODEL_DIR/log.txt
fairseq-train --task language_modeling \
    $DATA_BIN \
    --save-dir $MODEL_DIR \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    >$LOG 2>&1 & tail -f $LOG

# eval
DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline"
subset="test"
LOG=$MODEL_DIR/ppl_${subset}.txt
fairseq-eval-lm $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset $subset \
    > $LOG 2>&1 & tail -f $LOG


# save features
DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
MODEL="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best.pt"
#DATA_BIN="/data/yuxian/datasets/wikitext-103/data-bin"
#MODEL="/data/nfsdata2/nlp_application/models/language_models/adaptive_lm_wiki103.v2/model.pt"
for subset in "test" "valid" "train"; do
python eval_lm.py $DATA_BIN \
    --path $MODEL \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset $subset\
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap $DATA_BIN \
    --save-knnlm-dstore --dstore-fp16
done
