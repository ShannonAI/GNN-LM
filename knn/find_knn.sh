cd /userhome/yuxian/knnlm
#cd /home/mengyuxian/knnlm
export PYTHONPATH="$PWD"

#DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
DATA_BIN="/data/yuxian/datasets/wikitext-103/data-bin"


# build faiss indexes only for train
DS_DIRS=$DATA_BIN/train_dstore
metric="cosine"
#metric="l2"
#index="auto"
index="OPQ64_1024,IVF4096,PQ64"  # todo try more index
python knn/run_index_build.py \
  --dstore-dir $DS_DIRS \
  --index-type $index --chunk-size 1000000 \
  --metric $metric --use-gpu


# find knn index for train/valid/test
for subset in "valid" "test" "train"; do
  python knn/find_knn.py \
  --data-dir $DATA_BIN \
  --subset $subset \
  --cuda 0 --nprobe 32 --k 32
done

# quantize train features
index="OPQ64_1024,,PQ64"
python knn/quantize_features.py \
--data-dir $DATA_BIN  \
--subset "train" \
--chunk-size 10000000 \
--index $index --code-size 64 \
--compute-error  --use-gpu
