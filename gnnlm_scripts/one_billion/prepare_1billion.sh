#!/bin/bash

# download data
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

tar -xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled//news.en* >train.txt
cat 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en* >valid_and_test.txt
head -n 306688 valid_and_test.txt > valid.txt
tail -n 306688 valid_and_test.txt > test.txt


# truncate dataset to max-len 256
TEXT=/userhome/yuxian/data/lm/one-billion
for file in "valid.txt" "test.txt" "train.txt" ; do
  python remove_too_long_sentence.py \
  --input-file $TEXT/$file \
  --out-file $TEXT/$file.256 \
  --max-tokens 256
done


# download pretrained model
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2
tar -xvf adaptive_lm_gbw_huge.tar.bz2


# preprocess
TEXT=/userhome/yuxian/data/lm/one-billion
dict=/userhome/yuxian/train_logs/lm/one-billion/adaptive_lm_gbw_huge/dict.txt  # pretrained model dict
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.txt.256 \
    --validpref $TEXT/valid.txt.256 \
    --testpref $TEXT/test.txt.256 \
    --destdir $TEXT/data-bin-256 \
    --workers 12 \
    --srcdict $dict


# eval
DATA_BIN="/userhome/yuxian/data/lm/one-billion/data-bin-256"
MODEL_DIR="/userhome/yuxian/train_logs/lm/one-billion/adaptive_lm_gbw_huge"
LOG=$MODEL_DIR/ppl.txt
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/eval_lm.py $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
     --softmax-batch 1024 --log-interval 100 \
     --gen-subset test \
    > $LOG 2>&1 & tail -f $LOG


# save features
DATA_BIN="/userhome/yuxian/data/lm/one-billion/data-bin-256"
MODEL_DIR="/userhome/yuxian/train_logs/lm/one-billion/adaptive_lm_gbw_huge"
#for subset in "test" "valid" "train"; do
for subset in "train"; do
python eval_lm.py $DATA_BIN \
    --path $MODEL_DIR/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset $subset \
    --dstore-mmap $DATA_BIN \
    --save-knnlm-dstore --dstore-fp16 --first 1000000000  # we add this to preserve order of dataset
done
