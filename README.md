# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the python pandas library as pd
3. Read the contents of the Spam csv file
4. Display the first 5 rows of the dataset using head()
5. Assign x as v1 values and y as v2 values
6. From sklearn library select the feature extraction and import CountVectorizer

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NITHISH KUMAR S
RegisterNumber:212223240109  
*/
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:
![SVM For Spam Mail Detection](sam.png)

#### data.head()
![Screenshot 2024-05-08 185114](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/31d03f7f-f841-4d95-a6dd-ca9e551e6273)

#### data.tail()
![Screenshot 2024-05-08 185126](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/63914ac3-c90a-4b14-abe0-e7955590abaf)

#### data.info()
![Screenshot 2024-05-08 185138](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/1841e1bc-136f-49a8-8424-23be531f8144)

#### data.isnull().sum()
![Screenshot 2024-05-08 185147](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/35cfaac3-c204-437b-9771-12922b616b76)

#### Y_prediction value
![Screenshot 2024-05-08 185200](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/376958d3-2476-4073-957e-b721a30277c4)

#### Accuracy value
![Screenshot 2024-05-08 185209](https://github.com/nithish467/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150232274/cffcfdc4-eb1b-44c6-b21d-f14cbbbe99e3)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
