import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')


# 显示鸢尾花数据
def show_iris(mydataset):
    # 0. 查看数据集信息
    print("特征名称：", mydataset.data)

    # 1. 显示特征名称
    print("特征名称：", mydataset.feature_names)

    # 2. 把数据转换成 dataframe 格式，设置data， columns属性，目标值名称
    iris_d = pd.DataFrame(mydataset['data'], columns=mydataset.feature_names)
    # 再添加一列
    iris_d['species'] = mydataset.target
    print(iris_d)
    col1 = 'sepal length (cm)'
    col2 = 'petal width (cm)'

    # 3. sns.lmplot()显示
    # hue：颜色
    # fit_reg：是否画出回归线
    sns.lmplot(x=col1, y=col2, data=iris_d, hue='species', fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('iris')
    plt.show()
