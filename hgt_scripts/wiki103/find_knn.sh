#cd /userhome/yuxian/knnlm
#cd /home/mengyuxian/knnlm
cd /home/pkuccadmin/yuxian/knnlm # A100
export PYTHONPATH="$PWD"

#DATA_BIN="/userhome/yuxian/data/lm/wiki-103/data-bin"
#DATA_BIN="/userhome/yuxian/data/lm/wiki-103/resplit/data-bin"
DATA_BIN="/data/yuxian/wiki103-yunnao/data-bin"  # A100
#DATA_BIN="/data/yuxian/datasets/wikitext-103/data-bin"


# build faiss indexes only for train
DS_DIRS=$DATA_BIN/train_dstore
#DS_DIRS=$DATA_BIN/train_dstore-gcn_feat
metric="cosine"
#metric="l2"
index="OPQ64_1024,IVF4096,PQ64"  # todo try better indexes
python knn/run_index_build.py \
  --dstore-dir $DS_DIRS \
  --index-type $index --chunk-size 5000000 \
  --metric $metric --use-gpu


# find knn index for train/valid/test
#for subset in "valid" "test" "train"; do
for subset in "valid" "test" "valid1"; do
  python knn/find_knn.py \
  --data-dir $DATA_BIN \
  --subset $subset \
  --cuda 1 --nprobe 32 --k 1024
done

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


# (Optional) eval quantizer
CUDA_VISIBLE_DEVICES=0 python knn/eval_quantizer.py \
--data-dir $DATA_BIN  \
--use-gpu
