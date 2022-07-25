# EDS dataset and the code implementation of PSN
CVPR 2022 paper 《**Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network**》

## Download Link of EDS Dataset: 
Please go to the [webpage](https://github.com/DIG-Beihang/XrayDetection) and download according to the prompts.
<!--
```
(China mainland, BaiduNetdisk)：https://pan.baidu.com/s/1IzjPsoCowr2MYKbuOXuqUg (password：buaa)
(Other area, Google Drive): https://drive.google.com/file/d/17ids6mpIKpc_g67_CKC8aUnRDMeo6wxa/view?usp=sharing
```-->

## Prerequisites
- Python 3.6
- Pytorch 0.4.1
- CUDA 8.0 or higher
## Compile

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
## Testing
The `scripts` folder has all the testing scripts. For example, if you want to test a model trained from domain1 to domain2, just run:
```
sh scripts/test-all-1-2.sh
```

## Citation
If this work helps your research, please cite the following paper.
```
@inproceedings{Tao:CVPR22,
  author    = {Renshuai Tao and Hainan Li and Tianbo Wang and Yanlu Wei and Yifu Ding and Bowei Jin and, Hongping Zhi and Xianglong Liu and Aishan Liu},
  title     = {Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network},
  booktitle = {IEEE CVPR},
  year      = {2022},
  } 

```
