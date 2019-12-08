import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

df = pd.read_csv('nasa.csv')

df.isnull().sum()


df['Hazardous'].value_counts()
df['Hazardous'] = df['Hazardous'].map({True: 1, False: 0})

df['Close Approach Date'].value_counts()
del df['Close Approach Date']

df['Orbiting Body'].value_counts()
del df['Orbiting Body']

del df['Orbit Determination Date']

df['Equinox'].value_counts()
del df['Equinox']


corrmat = df.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 1, square = True)


## Baseline accuracy = 83.89%
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y,test_size = 0.2 , random_state = 0)

a = pd.DataFrame(y_test)
a[0].value_counts()
print("Baseline Accuracy for this Test Set = 84.54")

a = pd.DataFrame(y_train)
a[0].value_counts()
print('Baseline Accuracy for this Train Set = 83.72')
print()


def classifiers(clf, name):
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)
    cm_train = confusion_matrix(y_pred_train, y_train)
    
    print('Accuracy of {} for Test Set = {}'.format(name, (cm_test[1][1] + cm_test[0][0])/len(y_test)))
    print('Accuracy of {} for Train Set = {}'.format(name, (cm_train[1][1] + cm_train[0][0])/len(y_train)))
    print()
    
    return cm_train, cm_test



from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
cm_train, cm_test = classifiers(clf, 'Naive Bayes')


from sklearn.svm import SVC
clf = SVC(kernel = 'rbf')
clf.fit(X_train, y_train)
cm_train, cm_test = classifiers(clf, 'SVC')


from sklearn.tree import DecisionTreeClassifier as DTC
clf = DTC()
clf.fit(X_train, y_train)
cm_train, cm_test = classifiers(clf, 'Decision Tree')


import lightgbm as lgb
    
d_train = lgb.Dataset(X_train, label = y_train)
params = {}
clf = lgb.train(params, d_train, 100)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

for i in range(0, len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
        
for i in range(0, len(y_pred_train)):
    if y_pred_train[i] < 0.5:
        y_pred_train[i] = 0
    else:
        y_pred_train[i] = 1


from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(y_pred_train, y_train)

print('Accuracy of {} for Test Set = {}'.format('LightGBM', (cm_test[1][1] + cm_test[0][0])/len(y_test)))
print('Accuracy of {} for Train Set = {}'.format('LightGBM', (cm_train[1][1] + cm_train[0][0])/len(y_train)))
print()