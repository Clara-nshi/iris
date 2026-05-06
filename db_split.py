from sklearn.model_selection import train_test_split


# 数据集划分
def split_data(mydataset):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(mydataset.data, mydataset.target, test_size=0.3, random_state=22)
    print("数据总数量", len(mydataset.data))
    print("训练集中的x-特征值", len(X_train))
    print("测试集中的x-特征值", len(X_test))
    print("训练集中的y-目标值", len(y_train))
    print("测试集中的y-目标值", len(y_test))
    print(y_train)