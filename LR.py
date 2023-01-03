# -*- coding: utf-8 -*-
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def manual_separation(bad_line):
    right_split = bad_line[:-2] + [",".join(
        bad_line[-2:])]  # All the "bad lines" where all coming from the same last column that was containing ","
    return right_split


# Read the revievs
df = pd.read_csv('Reviews.csv', engine="python")

# ------ Show the data --------
# df
# df.info()
# df.describe()
# ------ Show the data --------

df['Helpful%'] = np.where(df['HelpfulnessDenominator'] > 0, df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
                          -1)
df['upvote%'] = pd.cut(df['Helpful%'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1],
                       labels=['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
print(
    df.groupby(['Score', 'upvote%']).agg({'Id': 'count'}))  # Bin labels must be one fewer than the number of bin edges
df_s = df.groupby(['Score', 'upvote%']).agg({'Id': 'count'}).reset_index()
pivot = df_s.pivot(index='upvote%', columns='Score')
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='g')

# Ignoring datapoints with score=3

df2 = df[df['Score'] != 3]
# Data Prepare
x = df2['Text']
y = df2['Score']
y_dict = {1: 0, 2: 0, 4: 1, 5: 1}
y = df2['Score'].map(y_dict)
labels = np.unique(y)
# - Bag of Words
c = CountVectorizer(stop_words='english')  # to ignore all english stopwords
# x_c = c.fit_transform(x)

# get train test
# x_train,x_test,y_train,y_test = train_test_split(x_c,y)

# Logistic Regression on Bag of Words
log = LogisticRegression(solver='liblinear')


# ml = log.fit(x_train, y_train)
# print(confusion_matrix(y_test, log.predict(x_test)))


# Automating NLP model and ML model
def text_fit(x, y, nlp_model, ml_model, coef_show=1):
    x_c = nlp_model.fit_transform(x)
    print('No. of features:{}'.format(x_c.shape[1]))
    x_train, x_test, y_train, y_test = train_test_split(x_c, y)

    ml = ml_model.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print("accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print("precision: {:.2f}%".format(metrics.precision_score(y_test, y_pred) * 100))
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format='')
    plt.show()


# tfidf = TfidfVectorizer(stop_words='english')
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
# text_fit(x,y,c,DummyClassifier())
text_fit(x, y, tfidf, log)

