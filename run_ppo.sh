#!/bin/bash
test_method="ppo"
train_method="olppo"
val_method="ppo"
ppo_total=100
ppo_sr=0.2
lr=0.00001
interval=5
amsgrad=True
weight_decay=1e-12
ema_decay=0
sampleloop=2
innerloop=5
batch_size=32
step_freq=2
minibatchnorm=False
vallike=True
# for run_times in `seq 1 1`:
# do
#     python main_generate.py -m general.test_method=$method general.seed=$RANDOM
# done
python main_generate.py -m dataset="ogbg-molppo" general.vallike=$vallike general.step_freq=$step_freq train.batch_size=$batch_size general.sampleloop=$sampleloop general.innerloop=$innerloop train.ema_decay=$ema_decay train.n_epochs=1250 train.weight_decay=$weight_decay train.amsgrad=$amsgrad general.val_check_interval=$interval general.val_method=$val_method general.train_method=$train_method general.ppo_total=$ppo_total general.ppo_sr=$ppo_sr train.lr=$lr train.batch_size=$batch_size general.seed=$RANDOM