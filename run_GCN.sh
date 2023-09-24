GPU=0
DATA=("DD" "PROTEINS" "PTC_MR" "IMDBBINARY" "FRANKENSTEIN")

for dataset in "${DATA[@]}";
do
  CUDA_VISIBLE_DEVICES=${GPU} python train_gcn.py --dataset $dataset --batch_size 32 
done

