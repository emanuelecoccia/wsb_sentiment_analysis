import numpy as np
import pandas as pd
import nltk.stem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

# loading the dataset
df_master = pd.read_csv('/content/wsb_dataset.csv')

df = df_master.copy()

# creating a list of stocks that we need to eliminate from the posts
# we do this to reduce bias: if GME posts are mostly bullish, the model will recognize the word 'GME' as bullish

# there are some stock symbols that are also normal words: we will leave them in the posts
not_symbols = ['DD', 'IS', 'A', 'BUY', 'ARE', 'ALL', 
              'GOOD', 'GAIN', 'WANT', 'HAS', 'KNOW',
              'GO', 'HOLD', 'AM', 'TOO', 'ONE', 'SEE', 'JOB',
              'IT', 'SEND', 'ATH', 'TALK', 'LOW', 'BY', 'SHIP',
              'REAL', 'IRL', 'PLUS', 'NEW', 'USD', 'YOLO', 'MUST',
              'U', 'PAYS', 'CEO', 'OLD', 'YOU', 'FOR', 'MOON', 'TELL',
              'WELL', 'ON', 'NOW', 'EVER', 'UP', 'AT', 'LIFE', 'JUST', 'BOIL',
              'FUND', 'LOAN', 'IMO', 'PT', 'BIG', 'BE', 'OPEN', 'PLAY', 'TA', 'IPO', 
              'FUD', 'BOND', 'CAN', 'FAIL', 'AWAY', 'FOMO', 'HEAR', 'VIEW', 'BEST', 'VERY',
              'MOVE', 'AIR', 'EAT', 'TRUE', 'DEEP', 'DAWN', 'RISE', 'CENT', 'LOVE', 'MIND', 'TEAM',
              'POST', 'COST', 'DATA', 'VOTE', 'STAY', 'NEXT', 'CASH', 'LACK', 'NEAR', 'COOL', 'ONCE', 'MASS',
              'SENT', 'HOPE', 'NEED', 'ELSE', 'PAY', 'APP', 'MAX', 'EPS', 'MID', 'OIL', 'VS', 'OR', 'MF', 'PLAN', 
              'PICK', 'EARN', 'INFO', 'SUM']

symbols = list(df.stock_symbol.unique())
symbols = list(set(symbols) - set(not_symbols))
symbols = sorted(symbols, key= len, reverse = True)

# this passage is needed to eliminate stocks substrings from the posts properly
long_symbols = [symbol for symbol in symbols if len(symbol) > 2]
short_symbols = [ ' {} '.format(symbol) for symbol in symbols if len(symbols) < 3]

# preparing the dataset

# we replace NaN values with an empty string ('')
df.replace(['[deleted]', '[removed]'], ['', ''], inplace = True)
df.fillna('', inplace = True)

# we merge the three predictors (title, body and flair) into one
df['text'] = df.title + ' ' + df.body + ' ' + df.flair
df.drop(columns = ['title', 'body', 'flair'], inplace = True)

# the model is not good at recognizing bearish posts because we lack bearish training data:
# it make more sense for our purpose to just distinguish between bullish and non-bullish posts
df['sentiment'].replace(['controversial', 'bearish', 'neutral'], 'not bullish', inplace = True)

# cleaning text
df.text = df.text.replace('$', '', regex = True)
df.text = df.text.replace('[^\w\s]', '', regex = True)
df.text = df.text.replace(long_symbols, '', regex = True)
df.text = df.text.replace(short_symbols, ' ', regex = True)

# it's better to shorten posts that are too long (usually Due Diligences)
# we do it by removing the central part
def removecentre(row):
  words = row.split()
  if len(words) > 500:
    row = words[:250] + words[-250:]
    row = ' '.join(row)
  return row 

df.text = df.text.apply(removecentre)

# stemming yields better results than lemmatizing
def stemrow(row):
  stemmer = nltk.stem.SnowballStemmer('english')
  words = row.split()
  new_row = []
  for word in words:
    new_row.append(stemmer.stem(word))
  new_row_str = ' '.join(map(str, new_row))
  return new_row_str

df.text = df.text.apply(stemrow)

'''
# we will comment out the lemmatization if needed for future projects
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def lemrow(row):
  lemmatizer = WordNetLemmatizer()
  words = row.split()
  new_row = []
  for word in words:
    new_word = lemmatizer.lemmatize(word, 'v')
    new_row.append(new_word)
  new_row_str = ' '.join(map(str, new_row))
  return new_row_str

df.text = df.text.apply(lambda x: x.lower())
df.text = df.text.apply(lemrow)
'''

# creating a pipeline
# using CountVectorizer to convert text documents to a matrix of token counts,
# with both monograms and bigrams as tokens
# using TfidTransformer to scale down the impact of tokens that occur very frequently
# using the SGDClassifier model to train the data

text_clf = Pipeline([
     ('vect', CountVectorizer(ngram_range=(1,2))),
     ('tfidf', TfidfTransformer()), #we could also use TfidfVectorizer and save one passage
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=0.00001, random_state = 2,
                           max_iter=6, tol=None, n_jobs = -1,
                           early_stopping = True, validation_fraction = .1
                           ))
])

# first we take only the labelled posts
training = df[df['is_train'] == True]
X, y = training['text'], training['sentiment']

# performing (stratified) K-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for train_index, test_index in skf.split(X, y):

  X_train = np.array(X.iloc[train_index]).flatten() # reshaping the input data
  X_test = np.array(X.iloc[test_index]).flatten()
  y_train = y.iloc[train_index]
  y_test = y.iloc[test_index]
    
  # fitting the model to the training data
  text_clf.fit(X_train, y_train)

  # predicting the test labels using the trained model on the test features
  predicted = text_clf.predict(X_test)

  # printing the metrics
  print(metrics.classification_report(y_test, predicted))
  
# final model, using both training and test data as training data
# (since we already know the metrics)
text_clf.fit(X, y)

# we use the model to label the other posts in the dataset
df_to_predict = df[df['is_train'] == False]
labels = text_clf.predict(df_to_predict['text'])

# lastly, we recompose the dataset with the labelled posts
df_final = df_master.copy()
df_final.loc[lambda x: df_final['is_train'] == False, 'sentiment'] = labels

# and we change the labels
df_final['sentiment'].replace(['controversial', 'bearish', 'neutral'], 'not bullish', inplace = True)

# exporting the dataset as a csv file
df_final.to_csv('wsb_dataset_final.csv', sep=';')
