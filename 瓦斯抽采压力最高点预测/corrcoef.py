import csv
import numpy as np

with open("预抽煤层瓦斯压力预测.csv", newline='', encoding='UTF-8-sig') as f:
    a = []  # 创建一个空列表用来整理数据
    b = []
    corr = []
    c = csv.reader(f)  # 读取csv文件里的数据
    for row in c:
        if row[0][0] == '渗':
            pass
        else:
            a.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])  # 将csv文件，逐行添加在a中
            b.append(row[7])
            corr.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
    a = np.asarray(a, dtype=float, order=None)  # 将a转换为numpy数组
    b = np.asarray(b, dtype=float, order=None)  # 将b转换为numpy数组
    corr = np.asarray(corr, dtype=float, order=None)
# X = df['sepal_length']
# Y = df['petal_length']
# result1 = np.corrcoef(X, Y)
result2 = np.corrcoef(corr, rowvar=False)
print(result2)
