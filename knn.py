import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split

import pandas as pd
df = pd.read_csv('dataset.csv')

df.replace("?",-99999,inplace=True)
df.drop(['id2'],1,inplace=True)

x = np.array(df.drop(['id1'],1))
y = np.array(df['id1'])

#sx = preprocessing.normalize(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print("Accuracy: ", accuracy)

example_measures = np.array([[1376316346.0,1.0,2599.999724,39.8666624347,1.53333333333,2097152.0,218100.8,0.0,7.0,0.666666666667,4.46666666667]])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)
