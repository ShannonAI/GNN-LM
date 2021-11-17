export PYTHONPATH="$PWD"

DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"

# build faiss indexes only for train
DS_DIRS=$DATA_BIN/train_dstore
#DS_DIRS=$DATA_BIN/train_dstore-gcn_feat
metric="cosine"
index="OPQ64_1024,IVF4096,PQ64"  # todo try better indexes
python knn/run_index_build.py \
  --dstore-dir $DS_DIRS \
  --index-type $index --chunk-size 5000000 \
  --metric $metric --use-gpu


# find knn index for train/valid/test
for subset in "valid" "test" "train"; do
  python knn/find_knn.py \
  --data-dir $DATA_BIN \
  --subset $subset \
  --cuda 1 --nprobe 32 --k 1024
done

# truncate knn neighbor to train with smaller k
for tgt_k in 8 16 64; do
python knn/truncate_neighbor_file.py \
--data $DATA_BIN --src-k 128 --tgt-k $tgt_k --subsets train valid test
done

# quantize train features
#index="OPQ64_1024,,PQ64"   # todo try better indexes
index="OPQ128_1024,,PQ128"
CUDA_VISIBLE_DEVICES=1 python knn/quantize_features.py \
--data-dir $DATA_BIN  \
--subset "train" \
--chunk-size 10000000 \
--index $index --code-size 128 \
--compute-error  --use-gpu

# convert pretrained ckpt to graph-ckpt
PRETRAINED="/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best.pt"
NEW_CKPT="/userhome/yuxian/train_logs/lm/wiki-103/qt_baseline_new/checkpoint_best.pt"
QUANTIZER=$DATA_BIN/quantizer
python fairseq_cli/convert_ckpt.py \
--ckpt $PRETRAINED \
--out $NEW_CKPT \
--quantizer $QUANTIZER


# (Optional) eval quantizer
CUDA_VISIBLE_DEVICES=0 python knn/eval_quantizer.py \
--data-dir $DATA_BIN  \
--use-gpu
