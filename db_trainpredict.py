# 导包
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_predict(mydataset):
    X_train, X_test, y_train, y_test = train_test_split(mydataset.data, mydataset.target, test_size=0.3, random_state=22)
    # 数据预处理，数据标准化
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    # 让测试集的均值和方法，转换测试集数据
    X_test = transfer.transform(X_test)
    # 机器学习（模型训练）
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(X_train, y_train)
    # 模型评估，直接计算准确率100个样本中模型预测对了多少
    myscore = estimator.score(X_test, y_test)
    print("准确率：", myscore)

    # 模型预测，需要对待预测数据，执行标准化
    print("通过模型查看分类类别-->", estimator.classes_)
    mydata = [[5.1, 3.5, 1.4, 0.2],
              [4.6, 3.1, 1.5, 0.2]]
    mydata = transfer.transform(mydata)
    print("mydata-->", mydata)
    mypred = estimator.predict(mydata)
    print("mypred-->", mypred)
    mypred = estimator.predict_proba(mydata)
    print("mypred-->", mypred)

    # 模型评估，利用sklearn.metrics包中的accuracy_score()方法
    y_predict = estimator.predict(X_test)
    myresult = accuracy_score(y_test, y_predict)
    print("myresult-->", myresult)


"""
1. 为了让被评估的模型更加准确可信，一般会使用交叉验证网格搜索去完成任务
2. 有些算法模型本身自带较多的超参数，无法高效的去筛选比较合适的超参数组合
3. 使用交叉验证和网格搜索可以提升模型的可信度和查找最佳参数组合的效率
"""

