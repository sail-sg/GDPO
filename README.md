# Graph Diffusion Policy Optimization
This paper introduces $\textit{graph diffusion policy optimization}$ (GDPO), a novel approach to optimize graph diffusion models for arbitrary (e.g., non-differentiable) objectives using reinforcement learning. GDPO is based on an $\textit{eager policy gradient}$ tailored for graph diffusion models, developed through meticulous analysis and promising improved performance. Experimental results show that GDPO achieves state-of-the-art performance in various graph generation tasks with complex and diverse objectives.
## Installing dependence
```
conda create --GDPO --file spec-list.txt
conda activate GDPO
pip install requrements.txt
```
If there are still issues, please refer to DiGress and add other dependencies as necessary.

In the following steps, make sure you have activated the GDPO environment.
```
conda activate GDPO
```
## Prepare Datasets
* ZINC250k and MOSES: [moldata](https://drive.google.com/file/d/1OlNGQCb-CrvUCF9jDVGyTNd78HuEG88q/view?usp=drive_link)

After downloading, unzip the files to the "dataset" folder, ensuring that the paths for ZINC250k and MOSES are "./dataset/zinc" and "./dataset/moses," respectively.

For Planar and SBM, they will be automatically downloaded during training.
## Prepare Pretrained Models
* Planar: [Planar Pretrained](https://drive.google.com/file/d/1jktMazwxjSb6jMEUSSYQmZ5V0JFNQdaS/view?usp=drive_link)
* SBM: [SBM Pretrained](https://drive.google.com/file/d/1KlJQ4H43q_IEMhvJO22vE1g22X2MjHjk/view?usp=drive_link)
* ZINC250k: [ZINC250k Pretrained](https://drive.google.com/file/d/1JGCKzh8KSLPyHk4gZS2TKgm8sdi7JBk4/view?usp=drive_link)
* MOSES: [MOSES Pretrained](https://drive.google.com/file/d/1eQJPPp_6QirfDisUIU1t4aBoepOWbl2l/view?usp=drive_link)

After downloading the pretrained models, place them in the "./pretrained" folder.

If you need to train your own pretrained models, please refer to the following commands and prepare the corresponding dataset as well as the config file (located in "./configs/experiment").
```
bash run_train.sh
```

If you are using your own pretrained models, you only need to change the "resume" field in the configuration file located in "./configs/experiment" to the address of your pretrained models (usually located in the "outputs" folder) during the fine-tuning phase.

## Run the toy experiments
In the paper, we designed a toy experiment that can be run without preparing any pretrained models.

The following command will run GDPO with 8 nodes.

```
bash run_ppo_toy.sh
```

If you want to change the number of nodes and the training method, such as running DDPO with 4 nodes, please modify the corresponding parameters in the "run_ppo_toy.sh" script.

The ".log" files named "evaluation" will display the corresponding evaluation results.

## Finetune with GDPO
Here, we mainly divide into two parts. For convenience in reproduction, we also directly provide the corresponding checkpoints. Please note that intermediate models during fine-tuning are saved in the "./multirun" folder.

### Planar and SBM

```
# finetune on the Planar with GDPO
bash run_ppo_planar.sh

# finetune on the SBM with GDPO
bash run_ppo_sbm.sh
```

Final model checkpoints:
* Planar: [Planar Final](https://drive.google.com/file/d/1u3mMInbnMKW7jRLn91MR8ceKyVziccLR/view?usp=drive_link)

* SBM: [SBM Final](https://drive.google.com/file/d/1uXh3NhYiJgokraYhEoxf3r0L-Nb5Qkvb/view?usp=drive_link)
### ZINC250k and MOSES
#### ZINC250k

```
#finetune on the ZINC250k with GDPO

bash run_ppo_prop.sh
```
This command will start fine-tuning on Zinc250k by default, targeting the 5ht1b protein. If you need to change the target protein, simply modify "+experiment=zinc_ppo_5ht1b.yaml" in "run_ppo_prop.sh" and replace "5ht1b" with the corresponding protein name. For example, "+experiment=zinc_ppo_parp1.yaml" will start fine-tuning targeting the parp1 protein.

We recommend running four or more different seeds for the same protein, i.e., running "bash run_ppo_prop.sh" four times, to mitigate the influence of random factors.

Final model checkpoints:
* parp1: [parp1 Final](https://drive.google.com/file/d/1oFCM16Gu_f2H0v8SvOqPsRTEm4sc-RLD/view?usp=drive_link)

* fa7: [fa7 Final](https://drive.google.com/file/d/1fitQT223-k9V3Fspxg4spD-1ncfbNE1G/view?usp=drive_link)

* 5ht1b: [5ht1b Final](https://drive.google.com/file/d/1vyirPjNg-XwRlmjHlojeKzLmZFxfkHN5/view?usp=drive_link)

* braf: [braf Final](https://drive.google.com/file/d/1fcz8LBqzUE_p1x_vE9dJgPZ5GU_bYaEB/view?usp=drive_link)

* jak2: [jak2 Final](https://drive.google.com/file/d/1-Elg_Uai0h4P77XkorIj8K2ch_zaLISy/view?usp=drive_link)

#### MOSES
```
#finetune on the MOSES with GDPO

bash run_ppo_moses.sh
```
The fine-tuning process on MOSES is essentially the same as on Zinc250k.

Final model checkpoints:
* parp1: [parp1 Final](https://drive.google.com/file/d/1bWqVMFj-ImiM84DFTLm7MfQeDSO3fBgY/view?usp=drive_link)

* fa7: [fa7 Final](https://drive.google.com/file/d/19_LLEn19IxbxKj-y4W_iwV8Wo56sFQf0/view?usp=drive_link)

* 5ht1b: [5ht1b Final](https://drive.google.com/file/d/1fZQChplyD2d5wGuz7QOw1EzqjmPZb9wv/view?usp=drive_link)

* braf: [braf Final](https://drive.google.com/file/d/1W8PzdnrLNSANeLve5VGv1QA0wMe8xtvK/view?usp=drive_link)

* jak2: [jak2 Final](https://drive.google.com/file/d/1aZ-czA6TcPKWg4tToEriwGyJ1hpcXeTt/view?usp=drive_link)

## Evaluation

### General Graph Evaluation
For Planar and SBM, modify line 353 in "main_generate.py" to specify "test_method" as "evalgeneral".

Then, modify the "test_only" variable in the "planar_test.yaml" and "sbm_test.yaml" files in "./configs/experiment" to point to the checkpoint path of the fine-tuned model.

Finally, run the following command:

```
# test model on Planar
bash run_test_graph.sh
```
If you want to test SBM, modify the "dataset" and "experiment" variables in "run_test_graph.sh".

### Molecular Graph Evaluation
For ZINC250k and MOSES, modify line 353 in "main_generate.py" to specify "test_method" as "evalproperty".

Modify the files in "./configs/experiment" to specify the "test_only" path.

Finally, run the following command:

```
# test model on Planar
bash run_test.sh
```
Modify the configuration in "run_test.sh" to evaluate different models with the target protein.

Note that we provide multiple final model checkpoints, but they are not in a format directly usable by PyTorch Lightning. To test these models, follow the code from lines 329-344 in "main_generate.py" to load these checkpoints into the model before conducting testing.
