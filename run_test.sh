#!/bin/bash 
test_method="evalproperty"

python main_generate.py -m dataset=$dataset +experiment=fa7_moses_test.yaml  general.target_prop=$target_prop general.seed=$RANDOM