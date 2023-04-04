# iris classification streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import plotly
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df['flower_name']=df.target.apply(lambda x:data.target_names[x]) 
st.title("Iris Classification")
st.write("This is a simple streamlit app to classify iris flowers into three classes")
st.write("The dataset used is the famous iris dataset")
st.write("The dataset contains 150 rows and 5 columns")
st.write("The columns are sepal length, sepal width, petal length, petal width and target")
st.write("The target column contains the class of the flower")
st.write("The classes are 0, 1 and 2")
st.write("The classes are setosa, versicolor and virginica respectively")
st.write("The dataset is available in the sklearn library")
st.write("The dataset is loaded into a pandas dataframe")
st.write("The dataframe is displayed below")
st.write(df.head())
st.write("The dataset is split into training and testing sets")
st.write("The training set is used to train the model")
st.write("The testing set is used to test the model")
st.write("The dataset is split into 80% training and 20% testing")
# split the dataset into training and testing sets
X=df.drop(['target','flower_name'],axis=1)
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
st.write("The dataset is scaled")
st.write("The dataset is scaled using the standard scaler")
# scale the dataset
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
st.write("The dataset is scaled")
st.write("The dataset is scaled using the standard scaler")
st.write("The scaled dataset is displayed below")
st.write(X_train)
st.write("The model is trained")
st.write("The model is trained using the logistic regression algorithm")
# train the model
model=LogisticRegression()
model.fit(X_train,y_train)
st.write("The model is trained")
st.write("The model is trained using the logistic regression algorithm")
st.write("The model is tested")
st.write("The model is tested using the testing set")
# test the model
y_pred=model.predict(X_test)
st.write("The model is tested")
st.write("The model is tested using the testing set")
st.write("The accuracy of the model is displayed below")
# accuracy of the model
st.write(accuracy_score(y_test,y_pred))
st.write("The confusion matrix of the model is displayed below")
# confusion matrix of the model
st.write(confusion_matrix(y_test,y_pred))
st.write("The classification report of the model is displayed below")
# classification report of the model
st.write(classification_report(y_test,y_pred))
st.write("The ROC curve of the model is displayed below")
# ROC curve of the model
# heatmap
st.write(sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)) 
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# plotly
import plotly.express as px
fig=px.scatter_3d(df,x='sepal length (cm)',y='sepal width (cm)',z='petal length (cm)',color='target')
st.plotly_chart(fig)
# shape
st.write('Shape of the dataset is',df.shape)
# describe
st.write('Description of the dataset is',df.describe())
# lmplot
st.write(sns.lmplot(x='sepal length (cm)',y='sepal width (cm)',data=df,hue='target',palette='Set1',fit_reg=False))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# sns pairplot
st.write(sns.pairplot(df,hue='target',palette='Set1'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# sns distplot
st.write(sns.distplot(df['sepal length (cm)'],kde=True,bins=30))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
