import csv
#引入CSV
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
#数据预处理
import numpy as np
from sklearn.model_selection import train_test_split
#引入划分训练集和测试集的工具
from sklearn import tree
#决策树


##数据处理
income_data = open(r'C:\Users\asus\Desktop\adult_income\income_data.csv', 'rt')
#在字符串中\是被当作转义字符来使用,所以要加r在前面
reader = csv.reader(income_data)
headers = next(reader)
#读取特征值的名称
featureList = []
labelList = []
#创建特征值和标签的列表
for row in reader:
    labelList.append(row[len(row)-1])
    #最后一列是标记的数值
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
#print(featureList)
#print(labelList)
#完成特征值列表和标记列表
vector = DictVectorizer()
#特征值0/1矩阵化
dummyX = vector.fit_transform(featureList).toarray()
labelY = preprocessing.LabelBinarizer()
#标记0/1化
dummyY = labelY.fit_transform(labelList)
income_matrixdata = np.column_stack((dummyX,dummyY))
np.savetxt('income_matrixdata.csv', income_matrixdata, delimiter = ',')
#创建训练集和测试集


#划分训练集和测试集
income_matrix=np.loadtxt(open(r'C:\Users\asus\Desktop\adult_income\income_matrixdata.csv', 'rt'),delimiter=",",skiprows=0)
#建立一个矩阵
X, y = income_matrix[:,:-1],income_matrix[:,-1]
#特征值和标签的取值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
#20%作为测试集，rm控制随机划分
train= np.column_stack((X_train,y_train))
np.savetxt('income_train.csv',train, delimiter = ',')
test = np.column_stack((X_test, y_test))
np.savetxt('income_test.csv', test, delimiter = ',')
#保存测试集和训练集


#决策树实现
income_train = np.loadtxt(open(r'C:\Users\asus\Desktop\adult_income\income_train.csv', 'rt'),delimiter=",",skiprows=0)
featureX, labelY = income_train[:,:-1],income_train[:,-1]
#特征值列表和标记列表的产生
classify = tree.DecisionTreeClassifier(criterion='entropy')
classify = classify.fit(featureX, labelY)
#除了熵，其他都是默认参数
#print("clf: " + str(clf))


#graphviz可视化决策树
with open(r'C:\Users\asus\Desktop\adult_income\adult_income.dot', 'w') as f:
    f = tree.export_graphviz(classify, feature_names=vector.get_feature_names(), out_file=f)
#创建一个dot文件用于graphviz绘图
    
    
#用测试集预测
predictedY = classify.predict(X_test)
#预测测试集数据
#print(str(predictedY))
accuracy_column = np.column_stack((y_test,predictedY))
#预测结果和实际结果合并
accuracy_number = 0
row_number = 0
for row in accuracy_column:
    row_number +=1
    if row[0] != row[-1]:
        accuracy_number +=1
#计算预测不符合的数据个数
accuracy = float(1-(accuracy_number/row_number))
print(float(accuracy))
#准确率