#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JZB32 热稳定性突变体预测脚本
使用训练好的 2D CNN 模型对酶突变体序列进行热稳定性预测。
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
from tqdm import tqdm

# -------------------- 固定参数 --------------------
MAX_LEN = 279                # 模型要求的序列长度（不足补 X）
PCA_COMPONENTS = 19          # 氨基酸特征降维后的维度
KERNEL_SIZES = [2, 2, 2, 3, 3, 2, 3]   # 对应 k_279
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 氨基酸特征加载 --------------------
def load_aa_features(aa_features_path):
    """加载氨基酸特征文件，返回 PCA 降维后的 DataFrame"""
    aa_df = pd.read_csv(aa_features_path, sep=',', encoding='GBK')
    # 假设第一列是氨基酸字母，其余为特征
    aa_index = aa_df.iloc[:, 0]   # 第一列作为索引
    features = aa_df.iloc[:, 1:].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=PCA_COMPONENTS)
    features_pca = pca.fit_transform(features_scaled)
    return pd.DataFrame(features_pca, index=aa_index)

# -------------------- 序列转矩阵 --------------------
def sequence_to_matrix(seq, aa_pca_df):
    """将蛋白质序列转换为模型输入的特征向量"""
    seq = seq.replace(' ', '')
    # 补 X 至固定长度
    seq = seq + 'X' * (MAX_LEN - len(seq))
    matrix = np.zeros((MAX_LEN, PCA_COMPONENTS))
    for i, aa in enumerate(seq):
        if aa in aa_pca_df.index:
            matrix[i] = aa_pca_df.loc[aa].values
        else:
            # 未知氨基酸用零向量代替
            matrix[i] = np.zeros(PCA_COMPONENTS)
    matrix = matrix.T   # 转置为 (19, 279)
    return matrix.flatten()

# -------------------- 模型定义 --------------------
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, PCA_COMPONENTS, MAX_LEN)

class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()
        self.reshape = Reshape()
        k = KERNEL_SIZES
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(19, k[0]), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, k[1]), stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, k[2]), stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(1, k[3]), stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(1, k[4]), stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(1, k[5]), stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 1 * k[6], 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 1 * KERNEL_SIZES[6])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------- 主预测函数 --------------------
def predict(input_file, output_file, model_dir, batch_size=3000):
    # 加载氨基酸特征
    aa_pca_df = load_aa_features(os.path.join(model_dir, 'AA_FEATURES.csv'))

    # 加载 scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

    # 加载模型
    model = LeNet(num_classes=2).to(DEVICE)
    checkpoint = torch.load(os.path.join(model_dir, '酶筛选模型.pth'), map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 读取输入序列（假设 CSV 包含一列名为 "Seq"）
    df = pd.read_csv(input_file, encoding='gbk')
    if 'Seq' not in df.columns:
        raise ValueError("输入 CSV 文件必须包含 'Seq' 列")
    sequences = df['Seq'].astype(str).tolist()

    # 分批预测
    results = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
        batch_seqs = sequences[i:i+batch_size]
        # 转换为特征矩阵
        X = []
        for seq in batch_seqs:
            vec = sequence_to_matrix(seq, aa_pca_df)
            X.append(vec)
        X = np.array(X)
        X = scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 类别1的概率（热稳定）
        results.extend(probs)

    # 保存结果
    out_df = pd.DataFrame({
        'Seq': sequences,
        'probability_stable': results
    })
    out_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"预测完成，结果已保存至: {output_file}")

# -------------------- 命令行接口 --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JZB32 热稳定性突变体预测')
    parser.add_argument('--input', '-i', required=True, help='输入 CSV 文件路径，需包含 "Seq" 列')
    parser.add_argument('--output', '-o', required=True, help='输出 CSV 文件路径')
    parser.add_argument('--model_dir', '-m', default='./model', help='存放模型文件的目录（包含 AA_FEATURES.csv, scaler.joblib, 突变酶模型.pth）')
    parser.add_argument('--batch_size', '-b', type=int, default=3000, help='预测批次大小，默认 3000')
    args = parser.parse_args()

    predict(args.input, args.output, args.model_dir, args.batch_size)