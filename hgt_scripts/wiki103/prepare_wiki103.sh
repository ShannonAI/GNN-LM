#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

URLS=(
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
)
FILES=(
    "wikitext-103-v1.zip"
)

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done

# preprocess
TEXT=/userhome/yuxian/data/lm/wiki-103  # yunnao
TEXT=/data/nfsdata2/nlp_application/datasets/corpus/english/wikitext-103  # gpu11
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir /data/nfsdata2/nlp_application/datasets/corpus/english/wikitext-103/data-bin \
    --workers 12

# train
#DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
#MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline"
DATA_BIN="/data/nfsdata2/nlp_application/datasets/corpus/english/wikitext-103/data-bin"
MODEL_DIR="/data/yuxian/train_logs/lm/wiki-103/fairseq_baseline"
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
#DATA_BIN="/data/nfsdata2/nlp_application/datasets/corpus/english/wikitext-103/data-bin"
#MODEL_DIR="/data/yuxian/train_logs/lm/wiki-103/urvashi"
LOG=$MODEL_DIR/ppl.txt
CUDA_VISIBLE_DEVICES=1 fairseq-eval-lm $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --tokens-per-sample 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset test \
    > $LOG 2>&1 & tail -f $LOG


# save features
for subset in "test" "valid" "train"; do
python eval_lm.py $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --tokens-per-sample 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset $subset \
    --dstore-mmap $DATA_BIN \
    --save-knnlm-dstore --dstore-fp16 --first 1000000000  # we add --first to preserve order of dataset
done
