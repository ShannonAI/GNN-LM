
cd /userhome/yuxian/data/lm/  # where you want to put your text

echo "- Downloading enwik8 (Character)"
if [[ ! -d 'enwik8' ]]; then
    mkdir -p enwik8
    cd enwik8
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python3 prep_enwik8.py
    cd ..
fi


# preprocess
#TEXT=/userhome/yuxian/data/lm/enwik8  # yunnao
TEXT=/data/nfsdata2/nlp_application/datasets/corpus/english/enwik8  # gpu11
DATA_BIN=$TEXT/data-bin  # yunnao
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir $DATA_BIN \
    --workers 12


# train NOTE(yuxian): here we have a hack to use pretrained transformer-xl here.
# Since transformer-XL is trained on latest fairseq, which is incompatible with knnlm,
# we firstly train a fake transformer model, then update its last layer's weight with pretrained transformer-xl
# and we do not do forward of transformer again in train/inference, but use the precompute features.
TEXT=/userhome/yuxian/data/lm/enwik8
DATA_BIN=$TEXT/data-bin  # yunnao
MODEL_DIR="/userhome/yuxian/train_logs/lm/enwik8/fairseq_fake"
mkdir -p $MODEL_DIR
CUDA_VISIBLE_DEVICES=0, fairseq-train --task language_modeling \
    $DATA_BIN \
    --save-dir $MODEL_DIR \
    --arch transformer_lm \
    --optimizer adam  \
    --lr 0.0003 --lr-scheduler fixed \
    --max-tokens 512 --update-freq 1 --tokens-per-sample 512 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test \
    --max-update 1
