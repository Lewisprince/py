import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
data_train = pd.read_csv('C:/Users/DELL/Desktop/advanced python  PPTS/train.csv') #导入数据
data_train.info()
data_train.describe()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数


plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u"获救情况 (1为获救)")  # 标题
plt.ylabel(u"人数")  # Y轴标签

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")  # 柱状图显示
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)  # 为散点图传入数据
plt.ylabel(u"年龄")  # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')  # 密度图
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")  # plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()  # 未获救
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()  # 获救
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()
fig = plt.figure()
plt.title(u'根据舱等级和性别的获救情况')

ax1 = plt.subplot2grid((2, 2), (0, 0))  # 将图像分为1行4列，从左到右从上到下的第1块
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                            label='female high class',
                                                                                            color='#FA2479')
ax1.set_xticklabels([u'获救', u'未获救'], rotation=0)
ax1.legend([u'女性/高级舱'], loc='best')

ax2 = plt.subplot2grid((2, 2), (0, 1))  # 将图像分为1行4列，从左到右从上到下的第2块
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                            label='female low class',
                                                                                            color='pink')
ax2.set_xticklabels([u'未获救', u'获救'], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = plt.subplot2grid((2, 2), (1, 0))
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                          label='male high class',
                                                                                          color='lightblue')
ax3.set_xticklabels([u'未获救', u'获救'], rotation=0)
plt.legend([u'男性/高级舱'], loc='best')

ax4 = plt.subplot2grid((2, 2), (1, 1))
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                          label='male low class',
                                                                                          color='steelblue')
ax4.set_xticklabels([u'未获救', u'获救'], rotation=0)
plt.legend([u'男性/低级舱'], loc='best')
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各登陆港口乘客的获救情况')
plt.xlabel(u'登陆港口')
plt.ylabel(u'人数')
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
x=np.arange(0,100)
data_train.Fare[data_train.Survived == 1].plot(kind='kde')
data_train.Fare[data_train.Survived == 0].plot(kind='kde')
plt.xlabel(u"票价")  # plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各票价的乘客获救密度分布")
plt.legend((u'获救', u'未获救'), loc='best')
plt.show()

g = data_train.groupby(['SibSp', 'Survived'])  # 数据分组
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
g = data_train.groupby(['Parch', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
fig = plt.figure()
fig.set(alpha=0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u' 有': Survived_cabin, u' 无': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u'按Cabin有无看获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()

# 将测试集导入，再将删除Survived数据的训练集与测验集进行合并，便于进行数据处理
data_test = pd.read_csv('C:/Users/DELL/Desktop/advanced python  PPTS/test.csv')  # 导入测验集数据

y = data_train['Survived']  # 将训练集Survived 数据存储在y中
del data_train['Survived']  # 删除训练集Survived数据
sum_id = data_test['PassengerId']  # 存储测试集乘客ID
df = pd.merge(data_train, data_test, how='outer')  # 合并无Survived数据的训练集与测验集，how = ‘outer’ 意为并集


# 缺失数据填充
#按Title填充年龄
data_test['Title']=data_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['Title']=df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df.Title.value_counts()
pd.crosstab(df.Title,df.Sex)
df[(df.Title=='Dr')&(df.Sex=='female')]
nn={'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}
df.Title=df.Title.map(nn)
#assign the female 'Dr' to 'Rarewoman'
df.loc[df.PassengerId==797,'Title']='Rarewoman'
df.Title.value_counts()
df[df.Title=='Master']['Sex'].value_counts()
df[df.Title=='Master']['Age'].describe()
df[df.Title=='Miss']['Age'].describe()
#由于miss年龄相差较大，定义年龄较小的miss为girl
df.Age.fillna(666,inplace=True)
def girl(aa):
    if (aa.Age!=666)&(aa.Title=='Miss')&(aa.Age<=14):
        return 'Girl'
    elif (aa.Age==666)&(aa.Title=='Miss')&(aa.Parch!=0):
        return 'Girl'
    else:
        return aa.Title
df['Title']=df.apply(girl,axis=1)
Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    df.loc[(df.Age==666)&(df.Title==i),'Age']=df.loc[df.Title==i,'Age'].median()
df.head()
df.groupby(['Title'])[['Age','Title']].mean().plot(kind='bar',figsize=(8,5))
plt.xticks(rotation=0)
plt.show()
#填充cabin
df.loc[df.Cabin.notnull(),'Cabin']=1
df.loc[df.Cabin.isnull(),'Cabin']=0
#用平均值填充Fare
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])  # 用数量最多项填充Embarked
pd.set_option('display.max_rows',None)
df['NameLength'] = df['Name'].apply(lambda x:len(x))
df['NameLength'] = df
df.info()
# 删掉无关因素
df = df.drop(['Name', 'PassengerId', 'Ticket','Title'], axis=1)  # 删除姓名、ID、船票信息、Title

# 将性别与港口用哑变量表示
dumm = pd.get_dummies(df[['Sex', 'Embarked']])  # '哑变量'矩阵
df = df.join(dumm)
del df['Sex']
del df['Embarked']

# 数据降维
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

# 训练模型
data_train = df[:len(data_train)]  # 将合并后的数据分离
data_test = df[len(data_train):]

X_train, X_val, y_train, y_val = train_test_split(data_train, y, test_size=0.3,
                                                  random_state=42)  # 以7：3（0.3）将训练集与获救结果随机拆分，随机种子为42

from sklearn.ensemble import GradientBoostingClassifier  # 引入Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
#进行训练
bagging=BaggingClassifier(LogisticRegression(C=0.06),n_estimators=100)
bagging.fit(X_train,y_train)
GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)  # 训练数据
RFC = RandomForestClassifier(n_estimators=500,random_state=6)
RFC.fit(X_train,y_train)
Log = LogisticRegression(C=0.06)
Log.fit(X_train,y_train)
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
DTC = DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_leaf=30)
DTC.fit(X_train, y_train)
#交叉验证并输出准确率
from sklearn.model_selection import cross_val_score
clfs = [bagging,GBC,RFC,MNB,Log,DTC]
kfold = 10
cv_results = []
for classifier in clfs:
    cv_results.append(cross_val_score(classifier,X_train.values,y_train.values,scoring="accuracy",cv=kfold))
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
ag = ['bagging','GBC','RFC','MNB','Log','DTC']
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors":cv_std,"Algorithm":ag})
for i in range(6):
    print(ag[i],cv_means[i])
#准确率取平均值作为总准确率
trainscore = (bagging.score(X_train,y_train)+GBC.score(X_train,y_train)+RFC.score(X_train,y_train)+MNB.score(X_train,y_train)+Log.score(X_train,y_train))/5
valscore = (bagging.score(X_val, y_val)+GBC.score(X_val, y_val)+RFC.score(X_val, y_val)+MNB.score(X_val, y_val)+Log.score(X_val, y_val))/5
print('训练集准确率：\n',trainscore)  # 分数
print('验证集准确率：\n',valscore)

# 预测测验集
pred1 = GBC.predict(data_test)  # pred 为预测结果
pred2 = bagging.predict(data_test)
pred3 = RFC.predict(data_test)
pred4 = MNB.predict(data_test)
pred5 = Log.predict(data_test)
pred6 = DTC.predict(data_test)
#六种方法预测结果取平均值并四舍五入作为最终结果
pred = (pred1+pred2+pred3+pred4+pred5+pred6)/6
pred = pred.astype(np.int)

pred = pd.DataFrame({'PassengerId': sum_id.values, 'Survived': pred})  # 格式化预测结果

pred.to_csv('pred.csv', index=None)  # 导出数据


