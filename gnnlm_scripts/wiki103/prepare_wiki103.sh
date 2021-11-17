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



# download pretrained model
MODEL_DIR="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline_20211115"
mkdir -p $MODEL_DIR
cd $MODEL_DIR
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2
tar -xvf adaptive_lm_wiki103.v2.tar.bz2
mv adaptive_lm_wiki103.v2/* ./
mv model.pt checkpoint_best.pt


# preprocess
TEXT=/userhome/yuxian/data/lm/wiki-103
DATA_BIN=$TEXT/data-bin-new
fairseq-preprocess \
    --only-source  \
    --srcdict $MODEL_DIR/dict.txt \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir $DATA_BIN \
    --workers 12


# eval
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
