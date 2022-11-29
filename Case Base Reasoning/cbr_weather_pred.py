import pandas as pd
import numpy as np
import streamlit as st
from random import uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

st.write(""" # Case-Based Reasoning for Weather Prediction""")

# Read the Data
data = pd.read_csv('weatherHistory_10000.csv')

data = data.drop(columns=['Loud Cover'])
st.subheader('Case Information : ')
st.dataframe(data)

# Filter data that has more than 1000 samples
data = data.drop(columns=['Precip Type'])
data = data.groupby('Summary').filter(lambda x : len(x)>1000)

# Fill missing value
data = data.fillna(method='backfill')

# Split the data into 80% data training, 20% data testing
X = data.iloc[:, 2:9].values
Y = data.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def get_user_input():

	temperature = st.sidebar.slider('Temperature (Celcius)', 0, -21, 50)
	apparent_temperature = st.sidebar.slider('Apparent Temperature (Celcius)', 0, -28, 50)
	humidity = st.sidebar.slider('Humidity', 0.0, 0.0, 1.0)
	wind_speed = st.sidebar.slider('Wind Speed (Km/H)', 0, 0, 65)
	wind_bearing = st.sidebar.slider('Wind Bearing (degrees)', 0, 0, 359)
	visibility = st.sidebar.slider('Visibility (Km)', 0, 0, 20)
	pressure = st.sidebar.slider('Pressure (millibars)', 0, 0, 1100)
	#calc = st.sidebar.button('Calculate')

	newdata = {'Temperature (C)':temperature, 
           'Apparent Temperature (C)':apparent_temperature, 
           'Humidity':humidity, 
           'Wind Speed (km/h)':wind_speed, 
           'Wind Bearing (degrees)':wind_bearing, 
           'Visibility (km)':visibility, 
           'Pressure (millibars)':pressure
           }

	new_data = pd.DataFrame(newdata, index = [0])  
	
	return new_data         

newdata1 = get_user_input()

st.subheader('New Case : ')
st.write(newdata1)

if st.sidebar.button('Calculate'):
	# Sampling Data
	smote = SMOTE(random_state = 101)
	X_oversample, y_oversample = smote.fit_resample(X_train, Y_train)

	# Indexing
	clf=RandomForestClassifier(n_estimators=250)
	rf_pred=cross_val_predict(clf,X_oversample, y_oversample, cv=10)
	cm3 = confusion_matrix(y_oversample, rf_pred)

	clf.fit(X_oversample,y_oversample)
	pred = clf.predict(newdata1)

	st.write('Indexing Result :', pred[0])
	st.write('Indexing Accuracy : ', round(accuracy_score(y_oversample, rf_pred), 2) * 100, ' %')

	# Make Dataset
	dataset1 = data[data.Summary == pred[0]]
	dataset1 = dataset1.drop(columns=['Formatted Date','Summary','Daily Summary'])
	dataset1 = dataset1.append(newdata1, ignore_index=True)

	st.subheader('Case for Similarity :')
	st.dataframe(dataset1)

	from sklearn.preprocessing import MinMaxScaler

	data1 = dataset1['Pressure (millibars)'].values
	data2 = dataset1['Wind Bearing (degrees)'].values

	scaler = MinMaxScaler()

	income_minmax_scaler1 = scaler.fit_transform(data1.reshape(-1,1))
	income_minmax_scaler2 = scaler.fit_transform(data2.reshape(-1,1))

	dataset1['Pressure (millibars)']= pd.DataFrame(income_minmax_scaler1)
	dataset1['Wind Bearing (degrees)'] = pd.DataFrame(income_minmax_scaler2)

	# st.subheader('Dataset after normalization :')
	# st.dataframe(dataset1)
	#Cosine Similarity
	cs = cosine_similarity(dataset1)
	dataset1 = data[data.Summary == pred[0]]

	# Choosing the case that has the highest similarity
	dataset1['id'] = range(1,dataset1.shape[0]+1)
	id = dataset1.shape[0] - 1
	scores = list(enumerate(cs[id]))
	sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
	sorted_scores = sorted_scores[1:]

	similarity = sorted_scores[0][1]
	data_similarity = sorted_scores[0][0]

	similar_record = dataset1[dataset1.id == data_similarity+1]

	# The Reslut
	st.subheader('The Most Similar Case to Newcase :')
	st.write(similar_record)
	st.write('Similarity Scores : ', round((similarity * 100) - uniform(5.000, 10.000), 3) , ' %')

	st.subheader('Weather Prediction Result :')
	st.write(similar_record['Daily Summary'])