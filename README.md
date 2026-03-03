# JZB32 热稳定性突变体预测模型

本仓库包含用于预测 **JZB32**（重组人截短型纤溶酶）突变体热稳定性的 2D CNN 模型及相关代码。模型基于氨基酸序列输入，输出突变体在 2–8°C 下稳定的概率，旨在辅助筛选适合室温储存的酶突变体。

## 模型简介

- 输入：蛋白质氨基酸序列（单字母编码），长度固定为 279（不足补 X）。
- 特征：每个氨基酸使用 19 维 PCA 降维特征表示（基于氨基酸的理化性质）。
- 架构：二维卷积神经网络（LeNet 风格），包含 6 个卷积层和 3 个全连接层。
- 输出：属于“热稳定类”（2–8°C）的概率，范围 [0,1]。



## 文件结构
├── predict.py # 预测脚本

├── requirements.txt # Python 依赖

├── README.md # 本文档

└── model/ # 存放模型文件的目录（需自行创建）

├── AA_FEATURES.csv # 氨基酸特征文件（GBK 编码）

├── scaler.joblib # 标准化器（joblib 格式）

└── 筛选酶模型.pth # 训练好的模型权重（PyTorch 格式）


## 安装依赖

建议使用 conda 或 virtualenv 创建独立环境，然后安装依赖：

pip install -r requirements.txt

## 1. 准备输入文件
输入应为 CSV 格式，至少包含一列名为 Seq，每行一个突变体序列（单字母氨基酸）。例如：
Seq
MYSFVN...
S111V
S111V/Q179R
...

## 2. 运行预测
python predict.py --input your_sequences.csv --output results.csv --model_dir ./model


## 3. 输出格式
输出文件为 CSV，包含两列：

Seq：原始序列

probability_stable：模型预测的该突变体在 2–8°C 稳定的概率

python predict.py -i data/single_mutations.csv -o data/predicted_results.csv -m ./model
