import numpy as np 
import pandas as pd

hart_attack_df=pd.read_csv(r"T:\VS_project\ML_Jadi\2nd_project\16325569497737558.csv")
#print(hart_attack_df.head())


#preprossesing

hart_attack_df=hart_attack_df[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','output']]
x=np.asarray(hart_attack_df[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']])
#print(x[:5])
y=np.asarray(hart_attack_df['output'])
#print(y[:5])

from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x)
#print(x[:5])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=3)
#print(f'train set={x_train.shape, y_train.shape}')
#print(f'test set={x_test.shape, y_test.shape}')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
#print(LR)

y_hat=LR.predict(x_test)
#print(y_hat)
#print(y_test)

y_hat_prob=LR.predict_log_proba(x_test)
print(y_hat_prob)

from sklearn.metrics import f1_score, classification_report
print(f1_score(y_test, y_hat))
print(classification_report(y_test, y_hat))