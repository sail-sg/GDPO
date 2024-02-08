#!/bin/bash //
train_method="gdpo"
val_method="ppo"
python main_generate.py -m dataset="zinc" +experiment=zinc_ppo_5ht1b.yaml general.train_method=$train_method general.val_method=$val_method general.seed=$RANDOM