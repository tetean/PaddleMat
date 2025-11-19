<h1>
<p align="center">
    <img src="properties/paddlemat-cover.png" alt="paddlemat logo" width="800"/>
</p>
</h1>

<h4 align="center">

[![Static Badge](https://img.shields.io/badge/PYTHON-3.9%2B-gray?style=for-the-badge&labelColor=blue)](https://python.org/downloads) 
&nbsp;&nbsp;&nbsp;&nbsp;
[![Static Badge](https://img.shields.io/badge/3.1.0%2B-grey?style=for-the-badge&label=PaddlePaddle&labelColor=purple)
](https://www.paddlepaddle.org.cn/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![Static Badge](https://img.shields.io/badge/%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%AD%A6%E9%99%A2-grey?style=for-the-badge&label=%E4%B8%AD%E5%B1%B1%E5%A4%A7%E5%AD%A6&labelColor=green)](https://cse.sysu.edu.cn/)

</h4>

---

# PaddleMat：基于PaddlePaddle框架的图神经网络的材料性质预测

本项目基于 **PaddlePaddle** 实现了 **ALIGNN（Atomistic Line Graph Neural Network）** 模型，并支持 **Evidential Uncertainty（证据不确定性估计）** 的回归任务，主要用于晶体材料性质预测（如带隙、形成能等）。

---

## 主要环境要求
```python
Python
PaddlePaddle
numpy
pandas
scikit-learn
PyYAML
tqdm
```

## 安装

1. 克隆项目
```python
git clone https://github.com/tetean/PaddleMat.git
cd PaddleMat
```

2. 创建环境与安装依赖

```python
conda env create -f environment.yml -n PaddleMat
conda activate PaddleMat
```



## 项目结构

```text
alignn-paddlepaddle/
│
├── train.py                    # 主训练脚本
├── alignn_paddle_model.py      # ALIGNN模型实现
├── alignn_data_utils.py        # 数据处理工具
├── config.yaml                 # 模型配置文件
├── environment.yml            # 依赖列表
│
├── utils/
│   ├── loss.py                # 损失函数
│   └── model_info.py          # 模型信息工具
│
├── data/                      # 数据目录
│   ├── task1/
│   ├── task2/
│   └── ...
│
├── checkpoints/               # 模型保存目录
└── logs/                     # 训练日志
```

## 数据集下载

为了快速读取和节省存储空间，我们提供了一个经过预处理的 JDFT2D 数据 pdstate 文件，该文件已打包，可供 ALIGNN 使用。

通过此链接下载：[https://drive.google.com/file/d/1IDTFJPrGv7gZ4cRvJTnr9MiPxf7vOwiM/view?usp=drive_link](https://drive.google.com/file/d/1IDTFJPrGv7gZ4cRvJTnr9MiPxf7vOwiM/view?usp=drive_link)

然后，把它放到`data/jdft2d文`件夹中。

## 配置模型参数
在 `config.yaml` 中配置不同的模型参数：

```python
Models:
    ALIGNN:
        model: "ALIGNN"
        cutoff: 8.0
        cutoff_extra: 3.0
        max_neighbors: 12
        use_canonize: True
        batch_size: 64
        epochs: 500
        lr: 0.001
        weight_decay: 0.00001
        optimizer: "AdamW"
        scheduler: "onecycle"
        evidential: "True"
        coeff: 0.01
        warmup_steps: 2000
        model_setting:
            alignn_layers: 4
            gcn_layers: 4
            atom_input_features: 92
            edge_input_features: 80
            triplet_input_features: 40
            embedding_features: 64
            hidden_features: 256
```

## 使用方法

1. 基础训练
```python
# 训练基础ALIGNN模型
python train.py \
    --config config.yaml \
    --data_dir /path/to/your/data \
    --task jdft2d \
    --model ALIGNN \
    --batch_size 64 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42
```

2. 启动不确定性量化训练
```python
# 修改config.yaml中evidential为"True"，然后训练
python train.py \
    --config config.yaml \
    --data_dir /path/to/your/data \
    --task jdft2d \
    --model ALIGNN \
    --batch_size 32 \
    --early_stop 50
```

## 训练命令详解

| 参数         | 类型  | 默认值       | 描述                 |
| ------------ | ----- | ------------ | -------------------- |
| `--config`   | str   | config.yaml  | 配置文件路径         |
| `--data_dir` | str   | 必需         | 数据根目录           |
| `--task`     | str   | 必需         | 任务名称             |
| `--model`    | str   | ALIGNN       | 模型名称             |
| `--batch_size` | int | 64           | 批大小               |
| `--train_ratio` | float | 0.8       | 训练集比例           |
| `--val_ratio` | float | 0.1         | 验证集比例           |
| `--seed`     | int   | 42           | 随机种子             |
| `--save_dir` | str   | checkpoints  | 模型保存目录         |
| `--early_stop` | int | 100          | 早停轮数（0 禁用）   |


## 输出

训练过程中会显示：
```python
Epoch 1/300
  Train Loss: 
  Val Loss:   
  Val MAE:    
  New best model! Saved to checkpoints/task/20241111-143022/ALIGNN_best.pdparams
--------------------------------------------------------------------------------
```

训练完成会显示：
```python
Test Results:
  Test Loss: 
  Test MAE:  
  (Best model from epoch 100)
```
训练完成后会生成：
```
1. ALIGNN_best.pdparams: 最佳模型权重 
2. ALIGNN_history.json: 训练历史记录 
3. 自动创建时间戳目录避免覆盖
```

---
> 欢迎提交Issue和Pull Request来改进项目！
> 如有问题或建议，请通过以下方式联系：
> ### a@tetean.com
