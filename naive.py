import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

df.replace("?",-99999,inplace=True)
df.drop(['id2'],1,inplace=True)


x = np.array(df.drop(['id1'],1))
y = np.array(df['id1'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Creating classifier and training
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#predict test results
y_pred = classifier.predict(x_test)

#print accuracy
print("Accuracy of Naive Bayes Classifier in % :  " + str(accuracy_score(y_test,y_pred)*100))
