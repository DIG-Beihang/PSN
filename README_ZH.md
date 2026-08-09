# EDS 数据集与 PSN 代码实现

CVPR 2022 论文《**Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network**》

[English](README.md) | [简体中文](README_ZH.md)

## EDS 数据集下载

请按照以下仓库中的说明获取 EDS 数据集下载链接：

https://github.com/DIG-Beihang/XrayDetection

## 跨域迁移学习

本仓库的一项核心特性是支持**不同 X 光成像域之间的跨域迁移学习**。用户无需手动修改训练流程，只需通过命令行参数 `--dataset` 和 `--datasets` 指定不同的源域与目标域，即可方便地构建跨域训练与迁移任务。

例如，可以在一个 X 光成像域上训练模型，并将其迁移到具有不同成像特性、设备类型或数据分布的另一个域：

```bash
python train.py \
    --dataset domain1 \
    --datasets domain2
```

通过修改 `--dataset` 和 `--datasets` 的取值，用户可以灵活构建不同的源域到目标域迁移设置，例如 `domain1 → domain2`、`domain1 → domain3`，以及数据集所支持的其他跨域组合。

不同域之间可能由于 X 光安检机型号、成像参数、目标外观以及采集环境等因素产生明显的数据分布差异，如下图所示：

<p align="center">
  <img src="figures/domain_difference.png" width="900">
</p>

<p align="center">
  <em>不同 X 光安检机成像域之间的分布差异示例。</em>
</p>

该设计使用户能够方便地在不同域设置下完成检测模型的训练、迁移与评测，为实际 X 光检测场景中的**跨域迁移、域适应与模型泛化能力研究**提供灵活支持。

## 环境依赖

- Python 3.6
- Pytorch 0.4.1
- CUDA 8.0 或更高版本

## 编译

```bash
pip install -r requirements.txt
cd lib
sh make.sh
```

## 训练

`scripts` 文件夹中包含所有训练脚本。例如，如果希望进行从 `domain1` 到 `domain2` 的跨域训练实验，可以直接运行：

```bash
sh scripts/train-1-2-fc.sh
```

## 测试

`scripts` 文件夹中包含所有测试脚本。例如，如果希望测试从 `domain1` 训练并迁移到 `domain2` 的模型，可以直接运行：

```bash
sh scripts/test-all-1-2.sh
```

## 引用

如果本工作对您的研究有所帮助，请引用以下论文：

```bibtex
@inproceedings{Tao:CVPR22,
  author    = {Renshuai Tao and Hainan Li and Tianbo Wang and Yanlu Wei and Yifu Ding and Bowei Jin and Hongping Zhi and Xianglong Liu and Aishan Liu},
  title     = {Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network},
  booktitle = {IEEE CVPR},
  year      = {2022},
}
```
