#importing modules
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#fetching data
x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' ]
nclasses = len(classes)

#Splitting the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)

#scaling 
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ", accuracy)

#sample images 
samples_per_class = 5
figure = plt.figure(figsize =(nclasses*2, (1+samples_per_class*2)))

idx_cls = 0
for cls in classes:
    idxs = np.flatnonzero(y==cls)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    i = 0
    for idx in idxs:
        plt_idx = i*nclasses + idx_cls +1
        p = plt.subplot(samples_per_class, nclasses, plt_idx);
        p = sns.heatmap(np.reshape(x[idx], (22,30)), cmap = plt.cm.gray, xticklabels= False,
            yticklabels=False, cbar=False);
        p = plt.axis('off');
        i+=1
    idx_cls+=1

#creating confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

p = plt.figure(figsize=(10,10));
p = sns.heatmap(cm, annot= True, fmt='d', cbar= False)