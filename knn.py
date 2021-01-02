# -*- coding: utf-8 -*-
"""knn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P1ewRbL9MXzGW2ZiwrRSu5PyrDCpXVzf
"""

import pandas as pd
dfx=pd.read_csv('/content/xdata.csv')
dfy=pd.read_csv('/content/ydata.csv')

dfx=dfx.iloc[:,1:3]
dfy=dfy.iloc[:,-1]

import numpy as np

x=dfx.values
y=dfy.values

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

q=np.array([2,3])
plt.scatter(x[:,0],x[:,1],c=y)
plt.scatter(q[0],q[1],color='red')
plt.show()
x.shape

def dist(x1,x2):
  return (sum((x2-x1)**2))

import numpy as np
def knn(x,y,q,k=5):
  val=[]
  m=x.shape[0]
  for i in range (0,m):
    d=dist(q,x[i])
    val.append([d,y[i]])

  val=sorted(val)
  val=val[:k]
  val=np.array(val)
  new=np.unique(val[:,1],return_counts=True)
  idx=new[1].argmax()
  pred=new[0][idx]
  print(val)
  return (pred)

val=knn(x,y,q)
val

plt.style.available