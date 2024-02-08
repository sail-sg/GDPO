#!/bin/bash /

train_method="gdpo"
val_method="ppo"
test_method="evalgeneral"
# for run_times in `seq 1 1`:
# do
#     python main_generate.py -m general.test_method=$method general.seed=$RANDOM
# done
python main_generate.py -m dataset="planar" +experiment=planar_test.yaml general.test_method=$test_method general.train_method=$train_method general.val_method=$val_method general.seed=$RANDOM