export PYTHONPATH="$PWD"

DATA_BIN="/userhome/yuxian/data/lm/one-billion/data-bin-256"

# build faiss indexes only for train
DS_DIRS=$DATA_BIN/train_dstore
metric="cosine"
index="OPQ16_64,IVF1048576,PQ16"  # for one-billion dataset to try2, which contains ~1B tokens
python knn/run_index_build.py \
  --dstore-dir $DS_DIRS \
  --index-type $index --chunk-size 100000  --max-train 20000000 \
  --metric $metric --use-gpu


# find knn index for train/valid/test
for subset in "valid" "test" "train"; do
  python knn/find_knn.py \
  --data-dir $DATA_BIN \
  --subset $subset \
  --cuda 1 --nprobe 32 --k 256 --bsz 16
done

# quantize train features
index="OPQ128_512,,PQ128"
CUDA_VISIBLE_DEVICES=1 python knn/quantize_features.py \
--data-dir $DATA_BIN  \
--subset "train" \
--chunk-size 10000000 \
--index $index --code-size 128 \
--compute-error  --use-gpu
