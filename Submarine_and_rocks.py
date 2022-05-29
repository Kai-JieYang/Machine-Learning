import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sub_data = pd.read_csv("C:/Users/User/Downloads/copy of sonar data.csv", header=None)
print(sub_data.groupby(60).mean())
X = sub_data.drop(columns = 60, axis=1)
Y = sub_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 0, stratify= Y)
print(X_train.shape)
clf = LogisticRegression()
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
acc = accuracy_score(pred, Y_test)


input_data = (0.0087,0.0046,0.0081,0.0230,0.0586,0.0682,0.0993,0.0717,0.0576,0.0818,0.1315,0.1862,0.2789,0.2579,0.2240,0.2568,0.2933,0.2991,0.3924,0.4691,0.5665,0.6464,0.6774,0.7577,0.8856,0.9419,1.0000,0.8564,0.6790,0.5587,0.4147,0.2946,0.2025,0.0688,0.1171,0.2157,0.2216,0.2776,0.2309,0.1444,0.1513,0.1745,0.1756,0.1424,0.0908,0.0138,0.0469,0.0480,0.0159,0.0045,0.0015,0.0052,0.0038,0.0079,0.0114,0.0050,0.0030,0.0064,0.0058,0.0030)
inn = np.asarray(input_data)
innt = inn.reshape(1, -1)
new_pred = clf.predict(innt)
print(new_pred)

