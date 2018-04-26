task=$1
CUDA_VISIBLE_DEVICES=$2 th main.lua -reward_epoch 12 -expIdx 1 -model rankerReader -task $1
cp ../trainedmodel/${task}1_best ../trainedmodel/${task}1_best_backup
CUDA_VISIBLE_DEVICES=$2 th main.lua -reward_epoch 12 -expIdx 1 -model rankerReader -task $1 -train_out 1
cd ..
sh preprocess.sh $1ans
cd main
CUDA_VISIBLE_DEVICES=$2 th main.lua -expIdx 1 -model reranker -task $1ans -pas_num 10 -max_epochs 10
