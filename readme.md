# 日志
用于记录项目进程和script介绍

## script介绍
dataset为使用的数据集，目前暂定steam-reviews和youtube-spam数据集，后续完善模型后考虑是否将Amazon-Product-Reviews纳入测试范围

dataloader用于进行数据加载、数据预处理、数据集划分等操作

checkpoints用于训练过程中存储模型参数

logs为实验日志

model为baseline模型

module为模型组件

## 待做
1. 增加LR/RNN/LSTM/GRU等上古模型
2. 使用Transformer-encoder进行分类
3. 使用mamba Block堆积木


