import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

st.title("News Classificaition")

df= pd.read_csv("cleaned_bbc.csv",usecols = ['category','cleaned'])
df

st.header('Enter News')

# Training model

log_regression = LogisticRegression()
vectorizer = TfidfVectorizer()
X = df['cleaned']
Y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression(random_state=1))])

# #Training model
model = pipeline.fit(X_train, y_train)

# #taking the data from user
# file = open('news.txt','r')
# news = file.read()
# file.close()


# creating a form
news = st.text_area("Enter News")
if st.button("Submit"):
    if news != "":
        news_data = {'predict_news':[news]}
        news_data_df = pd.DataFrame(news_data)
        predict_news_cat = model.predict(news_data_df['predict_news'])
        st.write("Predicted news category = ",predict_news_cat[0])
    else:
        st.write("There is no data  in File")

# st.button("Click me")





print()
