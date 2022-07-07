# -- coding: utf-8 --
"""
@author: SUBHRAJIT_DEY
"""
#importing libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math



# read CSV using Pandas
dataset = pd.read_csv('heart.csv')

#Function of Pandas
print(dataset.head())
print(dataset.shape)

#Get the column names of the file
print(dataset.columns)

#Find the number of mussing values
print(dataset.isnull().sum(axis = 0))
#We find the dataset has no missing values (COOL!) 

#Copying the dataset
dt = dataset.copy()


#Label Encoding
print(dt.dtypes)

cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
dt[cols] = dt[cols].astype('category')
print(dt.dtypes)


dt_cat = dt.copy()
    
#Dummy Variable Creation (Hot Encoding)
dt = pd.get_dummies(dt,drop_first=True)

#Extract the numerical columns
dt_copy = dt.copy()
scaler = MinMaxScaler()
cols_to_norm = ['Age','RestingBP','Cholesterol','MaxHR']
#cols_to_norm = ['Cholesterol','MaxHR']
dt_copy[cols_to_norm] = scaler.fit_transform(dt_copy[cols_to_norm])


#List of Values to be dropped


#Visualise(For seeing he data purposes)
# =============================================================================
# dt.hist(rwidth = 0.9)
# 
# plt.tight_layout()
# 
# =============================================================================



# =============================================================================
# plt.subplot(221)
# plt.title('Age vs Heart Disease')
# plt.scatter(dt_copy['HeartDisease'],dt_copy['Age'],s = 2, c='g')
# 
# plt.subplot(222)
# plt.title('RestingBP vs Heart Disease')
# plt.scatter(dt_copy['HeartDisease'],dt_copy['RestingBP'],s = 2,c='b')
# 
# plt.subplot(223)
# plt.title('Cholesterol vs Heart Disease')
# plt.scatter(dt_copy['HeartDisease'],dt_copy['Cholesterol'],s = 2,c='m')
# 
# plt.subplot(224)
# plt.title('MaxHR vs Heart Disease')
# plt.scatter(dt_copy['HeartDisease'],dt_copy['MaxHR'],s = 2,c='c')
# 
# plt.tight_layout()
# =============================================================================



# =============================================================================
# plt.subplot(221)
# plt.title('Age vs Heart Disease')
# plt.scatter(dt['HeartDisease'],dt['Age'],s = 2, c='g')
# 
# plt.subplot(222)
# plt.title('RestingBP vs Heart Disease')
# plt.scatter(dt['HeartDisease'],dt['RestingBP'],s = 2,c='b')
# 
# plt.subplot(223)
# plt.title('Cholesterol vs Heart Disease')
# plt.scatter(dt['HeartDisease'],dt['Cholesterol'],s = 2,c='m')
# 
# plt.subplot(224)
# plt.title('MaxHR vs Heart Disease')
# plt.scatter(dt['HeartDisease'],dt['MaxHR'],s = 2,c='c')
# 
# plt.tight_layout()
# 
# =============================================================================


#Checking the catagorical variables
# =============================================================================
# plt.subplot(231)
# cat_list = dt_cat['FastingBS'].unique()
# cat_average = dt_cat.groupby('FastingBS').mean()['HeartDisease']
# plt.bar(cat_list,cat_average)
# 
# 
# plt.subplot(232)
# cat_list = dataset['RestingECG']
# cat_average = dt['HeartDisease']
# plt.bar(cat_list,cat_average)
# 
# 
# 
# plt.subplot(233)
# cat_list = dt_cat['ExerciseAngina'].unique()
# cat_average = dt_cat.groupby('ExerciseAngina').mean()['HeartDisease']
# plt.bar(cat_list,cat_average)
# 
# plt.subplot(234)
# cat_list = dt_cat['ST_Slope'].unique()
# cat_average = dt_cat.groupby('ST_Slope').mean()['HeartDisease']
# plt.bar(cat_list,cat_average)
# 
# plt.subplot(235)
# cat_list = dt_cat['Sex'].unique()
# cat_average = dt_cat.groupby('Sex').mean()['HeartDisease']
# plt.bar(cat_list,cat_average)
# 
# =============================================================================


#No correlation
correlelation = dataset[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak','HeartDisease']].corr()



# =============================================================================
# 
# #Check the autocorrelation using acorr
# df1 = pd.to_numeric(dt_copy['HeartDisease'], downcast = 'float')
# plt.acorr(df1)
# #Not much autocorrelation Detected
# 
# 
# 
# =============================================================================


X = dt_copy.drop(['HeartDisease'],axis = 1)
Y = dt_copy[['HeartDisease']]


#Train Test Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state = 1234, stratify = Y)
#Graphs...................................................................................................



# =============================================================================
# 
# #Creating and train the sample multiple linear regression model
# hrt_reg = LinearRegression()
# # 
# # #Training or fit the training data
# hrt_reg.fit(X_train, Y_train)
# 
# 
# r2_train = hrt_reg.score(X_train, Y_train)
# r2_test  = hrt_reg.score(X_test, Y_test)
# 
# 
# # #Let's now predict the values of Y from test data
# Y_predict = hrt_reg.predict(X_test)
# 
# 
# 
# #Coefficient and Intercept of the regression line
# hrt_coefficient = hrt_reg.coef_
# hrt_intercept = hrt_reg.intercept_
# # #Coefficients for various parameters
# 
# 
# 
# # #How much error our model has made
# # #RMSE - Root Mean Squared Error
# hrt_rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))
# 
# 
# =============================================================================



#Creating Artificial Neural NetWorks for the datasets


model = Sequential()


model.add(Dense(48,
                 input_shape = (15,),
                 activation = 'relu',
                 kernel_initializer='RandomNormal'))

model.add(Dense(24,
                input_shape = (15,),
                activation = 'relu',
                kernel_initializer='RandomNormal'))

model.add(Dense(12,
                input_shape = (15,),
                activation = 'tanh',
                kernel_initializer='RandomNormal'))

model.add(Dense(6,
                input_shape = (15,),
                activation = 'tanh',
                kernel_initializer='RandomNormal'))

model.add(Dense(3,
                input_shape = (15,),
                activation = 'tanh',
                kernel_initializer='RandomNormal'))

model.add(Dense(1,
                input_shape = (15,),
                activation = 'sigmoid',
                kernel_initializer='RandomNormal'))

#Fit and Train the dataset

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 100, batch_size = 26)
#Graphs...................................................................................................

#Evaluate the model
accuracy_test = model.evaluate(X_test,Y_test)

#Confusion Matrix
Y_pred_prob = model.predict(X_test)


#Threshold 40%
predictions = (Y_pred_prob>0.4).astype('int32')
#..................................................................................................

deep_learning_rmse = math.sqrt(mean_squared_error(Y_test, predictions))



#Confusion Matrix
cm = confusion_matrix(Y_test,predictions)



#Convertion of the model into tensorflow lite

KERAS_MODEL_NAME = 'tf_heart_pred.h5'
model.save(KERAS_MODEL_NAME)

# =============================================================================
tf_model = tf.keras.models.load_model('tf_heart_pred.h5')
# 
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# 
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 
tflite_model = converter.convert()
# 
open('tf_heart_pred.tflite','wb').write(tflite_model)



# =============================================================================
# 
# 
# 
# 
# tflite_interpreter = tf.lite.Interpreter(model_path='tf_heart_pred.tflite')
# 
# input_details = tflite_interpreter.get_input_details()
# output_details = tflite_interpreter.get_output_details()
# 
# print("== Input details ==")
# print("name:", input_details[0]['name'])
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])
# 
# print("\n== Output details ==")
# print("name:", output_details[0]['name'])
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])
# 
# 
# 
# 
# 
# tflite_interpreter.resize_tensor_input(input_details[0]['index'], (230,15))
# tflite_interpreter.resize_tensor_input(output_details[0]['index'], (230,1))
# tflite_interpreter.allocate_tensors()
# 
# input_details = tflite_interpreter.get_input_details()
# output_details = tflite_interpreter.get_output_details()
# 
# print("== Input details ==")
# print("name:", input_details[0]['name'])
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])
# 
# print("\n== Output details ==")
# print("name:", output_details[0]['name'])
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])
# tflite_interpreter.set_tensor(input_details[0]['index'], X_test)
# 
# tflite_interpreter.invoke()
# 
# tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
# print("Prediction results shape:", tflite_model_predictions.shape)
# 
# 
# 
# 
# =============================================================================


# Load TFLite model and allocate tensors.

interpreter = tf.lite.Interpreter(model_path="tf_heart_pred.tflite")
interpreter.allocate_tensors()
 
# # Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
 
#Resize Test size of tensor
interpreter.resize_tensor_input(input_details[0]['index'], (230,15))
interpreter.resize_tensor_input(output_details[0]['index'], (230,1))
interpreter.allocate_tensors()
#Change...................................................................................................
# 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#  
# 
# Test model on random input data.
input_shape = input_details[0]['shape']
# 
X_test1 = X_test.copy()
# =============================================================================



#Convert all independent variables to float32
X_test['Oldpeak'] = X_test['Oldpeak'].astype('float32')
X_test['Age'] = X_test['Age'].astype('float32')
X_test['RestingBP'] = X_test['RestingBP'].astype('float32')
X_test['Cholesterol'] = X_test['Cholesterol'].astype('float32')
X_test['FastingBS'] = X_test['FastingBS'].astype('float32')
X_test['MaxHR'] = X_test['MaxHR'].astype('float32')
X_test['Sex_M'] = X_test['Sex_M'].astype('float32')
X_test['ChestPainType_ATA'] = X_test['ChestPainType_ATA'].astype('float32')
X_test['ChestPainType_TA'] = X_test['ChestPainType_TA'].astype('float32')
X_test['ChestPainType_NAP'] = X_test['ChestPainType_NAP'].astype('float32')
X_test['RestingECG_Normal'] = X_test['RestingECG_Normal'].astype('float32')
X_test['RestingECG_ST'] = X_test['RestingECG_ST'].astype('float32')
X_test['ExerciseAngina_Y'] = X_test['ExerciseAngina_Y'].astype('float32')
X_test['ST_Slope_Up'] = X_test['ST_Slope_Up'].astype('float32')
X_test['ST_Slope_Flat'] = X_test['ST_Slope_Flat'].astype('float32')

 
types = X_test.dtypes
 
 
 
interpreter.set_tensor(input_details[0]['index'], X_test)
interpreter.invoke()


#Testing the Model
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


#Threshold = 40%
predictions1 = (output_data>0.4).astype('int32')
#..................................................................................................


#Confusion Matrix of tflite
cm_lite = confusion_matrix(Y_test,predictions1)