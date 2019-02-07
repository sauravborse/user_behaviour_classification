import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('dataset.csv')

df.replace("?",-99999,inplace=True)
df.drop(['id2'],1,inplace=True)

x = np.array(df.drop(['id1'],1))
y = np.array(df['id1'])




x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example_measures = np.array([[1376314846,1,2599.999724,34.6666629867,1.33333333333,2097152,127225.333333,0,4.6,0,0.2]])

example_measure = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)