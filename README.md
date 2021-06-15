# Counterfactual Off-Policy Training for Neural Dialogue Generation

A pytorch implementation for [Counterfactual Off-Policy Training for Neural Dialogue Generation](https://www.aclweb.org/anthology/2020.emnlp-main.276/)

![](https://github.com/qfzhu/COPT/blob/master/model.png)
 
## Requirements

python 2.7

opennmt 0.3

## Quickstart

### Prepare the data

Download the data from the following link, and put it under the root of the project.

```bash
https://drive.google.com/drive/folders/1IDjn5f7mILBCAsfbqyLwzGdWRKiGigxO?usp=sharing
```

### Pre-Train G

```bash
python train.py -data data/daily -save_model model/daily -word_vec_size 300 -dropout 0.2 -gpu 1 -epochs 15 -training_mode pre_g -pre_word_vecs_enc data/daily.emb.enc.pt -pre_word_vecs_dec data/daily.emb.dec.pt
```

### Pre-Train D

This is optional according to the adversarial learning model that COPT applied to. For example, StepGAN does not need this step.

```bash
python train.py -data data/daily -gpu 1 -training_mode pre_d -epochs 20 -train_from checkpoint
```

### Adversarial Learning

```bash
python train.py -data data/daily -gpu 1 -training_mode adv -epochs 25 -train_from checkpoint
```

### Inference

```bash
python translate.py -src src-test.txt -tgt src-test.txt -ref src-test.txt -verbose -gpu 1 -model checkpoint
```

## Citation

```
@inproceedings{copt,
  author    = {Qingfu Zhu and
               Weinan Zhang and
               Ting Liu and
               William Yang Wang},
  title     = {Counterfactual Off-Policy Training for Neural Dialogue Generation},
  booktitle = {Proc. EMNLP},
  year      = {2020}
}
```
