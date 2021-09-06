import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("spam.csv")
# print(df.head(5))

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
# print(df.head())

df.rename(columns={'v1':'labels', 'v2':'message'}, inplace='True')

# print(df.shape)
df.drop_duplicates(inplace=True)
# print(df.shape)
df['labels'] = df['labels'].map({'ham':0, 'spam':1})
# print(df)

def clean_data(message):
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

# test = clean_data('hello, how are ?! + will you go to school ?')
# print(test)

df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

# print(x)

cv = CountVectorizer()

x = cv.fit_transform(x)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = MultinomialNB().fit(x_train, y_train)

# predictions = model.predict(x_test)

# print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))

def predict(text):
    text = [clean_data(text)]
    x = cv.transform(text).toarray()
    p = model.predict(x)
    if(p==[1]):
        return 'It looks like: Spam'
    else:
        return 'It looks like: Not Spam'

# print(predict('Congratulations, You won a lottery of $3000'))

st.title('Spam Classifier')
st.image('spam.jpg')

user_input = st.text_input('Write your message here')
submit = st.button('Predict')

if submit:
    answer = predict(user_input)
    st.text(answer)
    







