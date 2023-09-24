GPU=0
DATA=("PTC_MR" "IMDBBINARY" "FRANKENSTEIN")

for dataset in "${DATA[@]}";
do
  CUDA_VISIBLE_DEVICES=${GPU} python train_gcn.py --dataset $dataset --batch_size 32
done

