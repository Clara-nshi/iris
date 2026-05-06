"""
1. 获取数据集
2. 数据基本处理
3. 特征工程
4. 机器学习（模型训练）
5. 模型评估

使用可视化加载和探索数据，以确定特征是否能将不同类别分开
通过标准化特征，并随机抽样到训练集和测试机来准备数据
通过统计学，利用准确率评估机器学习模型
"""
# 导包
# 显示属性
from db_attribute import ds_attributes
# 加载鸢尾花数据集
from db_showiris import show_iris
# 数据集划分
from db_split import split_data
# 数据标准化+模型训练+模型评估+模型预测
from db_trainpredict import train_predict
# 1. 获取数据集
from sklearn.datasets import load_iris

# 调用db_attribute，显示属性
ds_attributes(load_iris())
# 加载鸢尾花数据集(图形)
show_iris(load_iris())

# 2. 数据基本处理
# 数据集划分
split_data(load_iris())

# 3. 数据集预处理-数据标准化
# 4. 机器学习（模型训练）
# 5. 模型评估
# 6. 模型预测
train_predict(load_iris())




