# CLBLP-2023

## Bag of Words and TF-IDF

### Code Snippets

```
sentences = [
    "আমি বাংলায় গান গাই",
    "আমি বাংলার গান গাই",
    "আমি আমার আমিকে চিরদিন এই বাংলায় খুঁজে পাই"
]

```

```
# Save mapping on which index refers to which words
col_map = {v:k for k, v in cv.vocabulary_.items()}
# Rename each column using the mapping
for col in cv_corpus.columns:
    cv_corpus.rename(columns={col: col_map[col]}, inplace=True)
cv_corpus
```

```
# Convert sparse matrix to dataframe
tf_corpus = pd.DataFrame.sparse.from_spmatrix(tf_corpus)
# Save mapping on which index refers to which words
col_map = {v:k for k, v in tf_idf.vocabulary_.items()}
# Rename each column using the mapping
for col in tf_corpus.columns:
    tf_corpus.rename(columns={col: col_map[col]}, inplace=True)
tf_corpus
```


Dataset:
```
!wget -O Emotion.csv https://www.dropbox.com/s/pjzwlxbmm7sq4sf/emotion.csv?dl=0
```

```

%%time
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
import re
import nltk
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
np.random.seed(42)
```

### Data Cleaning

```

%%time
from bnlp.corpus import stopwords, punctuations
from nltk import word_tokenize
## Data samples before cleaning
for i in range(10):
  print(data['TEXT'][i])
## Cleaning the data. Removing newlines, unnecessary symbols

punctuations = '''“”!()-[]{};:'"\,<>./?@#$%^&*_~�।’‘1234567890''' #we can add suitable extra punctuation all the time
def remove_punctuation(d):
    review = d.replace('\n', '')
    no_punct = ""
    for char in review:
      if char not in punctuations:
         no_punct = no_punct + char
    return no_punct

def remove_stopwords(d):
  text_tokens = word_tokenize(d)
  bn_stopwords = stopwords() 
  tokens_without_sw = [word for word in text_tokens if not word in bn_stopwords]
  ls = ""
  for w in tokens_without_sw:
    ls = ls +" "+w
  return ls

data['cleaned'] = data['TEXT'].apply(remove_punctuation)

```

Result printing
```
def print_report(pipe):
    y_pred = pipe.predict(X_test)
    report = metrics.classification_report(y_test, y_pred,
        target_names=class_names)
    cm= confusion_matrix(y_test, y_pred)
    print(report)
    print(cm)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
```

