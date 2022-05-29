import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

diabete = pd.read_csv("C:/Users/User/Downloads/diabetes.csv")
X = diabete.drop(columns="Outcome", axis=1)        
Y = diabete["Outcome"]
scaler =  StandardScaler()
new_data =  scaler.fit_transform(X)
X = new_data
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
clf = svm.SVC(kernel= "linear")
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
acc = accuracy_score(pred, y_test)
print(acc)

haha  = (5,116,74,0,0,25.6,0.201,30)
input_data = np.asarray(haha)
inn = input_data.reshape(1,-1)
std_data = scaler.transform(inn)
k = clf.predict(std_data)
if k[0] == 1:
    print("Diabete")
else:
    print("Not Diabete")
