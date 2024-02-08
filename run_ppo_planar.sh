#!/bin/bash //
train_method="gdpo"
val_method="ppo"
lr=1e-5
innerloop=1
sampleloop=2
step_freq=1
fix=0.5
# for run_times in `seq 1 1`:
# do
#     python main_generate.py -m general.test_method=$method general.seed=$RANDOM
# done
python main_generate.py -m dataset="planar" +experiment=planar_ppo.yaml general.fix=$fix train.lr=$lr general.train_method=$train_method general.val_method=$val_method general.step_freq=$step_freq general.innerloop=$innerloop general.sampleloop=$sampleloop  general.seed=$RANDOM