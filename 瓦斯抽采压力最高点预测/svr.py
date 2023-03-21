import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import train_test_split
import csv

with open("预抽煤层瓦斯压力预测.csv", newline='', encoding='UTF-8-sig') as f:
    a = []  # 创建一个空列表用来整理数据
    b = []
    c = csv.reader(f)  # 读取csv文件里的数据
    for row in c:
        if row[0][0] == '渗':
            pass
        else:
            a.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])  # 将csv文件，逐行添加在a中
            b.append(row[7])
    a = np.asarray(a, dtype=float, order=None)  # 将a转换为numpy数组
    b = np.asarray(b, dtype=float, order=None)  # 将b转换为numpy数组

with open("鹤煤八矿真实数据1.csv", newline='', encoding='UTF-8-sig') as f1:
    a1 = []  # 创建一个空列表用来整理数据
    b1 = []
    c1 = csv.reader(f1)  # 读取csv文件里的数据
    for row in c1:
        if row[0][0] == '渗':
            pass
        else:
            a1.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])  # 将csv文件，逐行添加在a中
            b1.append(row[7])
    a1 = np.asarray(a1, dtype=float, order=None)  # 将a转换为numpy数组
    b1 = np.asarray(b1, dtype=float, order=None)  # 将b转换为numpy数组
    print(a1, b1)


###########1.数据生成部分##########
def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 3 + 0.1 * x1
    return y


x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.25, random_state=0)


###########2.回归部分##########
def try_different_method(model):
    model.fit(x_train, y_train)
    score = model.score(a1, b1)
    result = model.predict(a1)
    plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(result[2000::])), y_test[2000::], 'go-', label='true value')
    plt.plot(np.arange(len(result)), b1, 'go-', label='true value')
    # plt.plot(np.arange(len(result[2000::])), result[2000::], 'ro-', label='predict value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model

model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm

model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor

model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor

model_ExtraTreeRegressor = ExtraTreeRegressor()
###########4.具体方法调用部分##########
# try_different_method(model_DecisionTreeRegressor)
# try_different_method(model_LinearRegression)
# try_different_method(model_SVR)
# try_different_method(model_KNeighborsRegressor)
# try_different_method(model_RandomForestRegressor)
# try_different_method(model_AdaBoostRegressor)
# try_different_method(model_GradientBoostingRegressor)
# try_different_method(model_BaggingRegressor)
try_different_method(model_ExtraTreeRegressor)
