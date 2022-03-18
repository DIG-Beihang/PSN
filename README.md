# PSN
Implement of CVPR 2022 paper 《**Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network**》

## Datasets

## Prerequisites
- Python 3.6
- Pytorch 0.4.1
- CUDA 8.0 or higher
## Compilation

```
pip install -r requirements.txt
cd lib
sh make.sh
```

## Training
The `scripts` folder has all the training scripts. For example, if you want to train an experiment from domain1 to domain2, just run:
```
sh scripts/train-1-2-fc.sh
```
## Test
The `scripts` folder has all the testing scripts. For example, if you want to test a model trained from domain1 to domain2, just run:
```
sh scripts/test-all-1-2-.sh
```

## Citation
If this work helps your research, please cite the following paper.
```

```