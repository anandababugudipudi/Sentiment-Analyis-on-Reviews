# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:33:01 2021

@author: anand
"""
##############################################################################
# Import necessary packages
##############################################################################
print("\nImporting necessary packages...")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pickle

##############################################################################
# Read the data from CSV file into DataFrame
##############################################################################
print("\nReading the data from CSV file into DataFrame...")
df =  pd.read_csv('balanced_reviews.csv')

##############################################################################
# Data Preprocessing 
##############################################################################
print("\nApplying Data Preprocessing Techniques on Data...")
# Check for Null Values
df.isnull().any(axis = 0)
# Drop the rows with missing values
df.dropna(inplace =  True)
# Drop the rows with rating 3 and keeping the remaining
df = df [df['overall'] != 3]
# Creating a new label 'Positivity' based on the values in overall column
df['Positivity'] = np.where(df['overall'] > 3 , 1 , 0)

##############################################################################
# Separating Features (X) and labels (y)
##############################################################################
print("\nSeparating Features (X) and labels (y)...")
X = df['reviewText']
y = df['Positivity']

##############################################################################
# Split the Features and labels into Train and Test Data
##############################################################################
print("\nSplitting the Features and labels into Train and Test Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42 )

##############################################################################
# Apply NLP Techniques on reviews
##############################################################################
print("\nApplying NLP Techniques on Data...")
# Apply TfidfVectorizer on train features
vect = TfidfVectorizer(min_df = 5).fit(X_train)
X_train_vectorized = vect.transform(X_train)

##############################################################################
# Build a Logistic Regression Model
##############################################################################
print("\nBuilding the Logistic Regression Model...")
# Build the model
model = LogisticRegression(max_iter = 1000)
# Train the model
print("\nTraining the Model...")
model.fit(X_train_vectorized, y_train)
# Get predictions on test data
predictions = model.predict(vect.transform(X_test))

##############################################################################
# Evaluate the model
##############################################################################
print("\nEvaluating the Model...")
# Get the confusion matrix
cm = confusion_matrix(y_test, predictions)
# Get the ROC_AUC Score
auc = roc_auc_score(y_test, predictions)

##############################################################################
# Pickle the model and vocabulary for using in project
##############################################################################
print("\nPickling the Model and Vocabulary...")
# Pickle the model
model_file  = open("pickle_model.pkl","wb")
pickle.dump(model, model_file)
# Pickle the vocabulary
vocab_file = open('features.pkl', 'wb')
pickle.dump(vect.vocabulary_, vocab_file)

print("\nSuccessfully Completed all the Tasks...")