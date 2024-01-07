import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from joblib import dump

# loading the datasets
train_df = pd.read_csv('/train_set.csv')
test_df = pd.read_csv('/test_set.csv')

# splitting the datasets into features and labels
X_train, y_train = train_df['Job_offer'], train_df['Label']
X_test, y_test = test_df['Job_offer'], test_df['Label']

# it is better to feed the model with numbers,
# therefore we change the labels into floating numbers
labels = {'Java Developer':0, 'Software Engineer':1, 'Programmer':2, 'System Analyst':3, 'Web Developer':4}
y_train = np.array([labels[label] for label in y_train], dtype=np.float32)
y_test = np.array([labels[label] for label in y_test], dtype=np.float32)

# creating a pipeline
# using CountVectorizer to convert text documents to a matrix of token counts,
# with both monograms and bigrams as tokens
# using TfidTransformer to scale down the impact of tokens that occur very frequently
# using the SGDClassifier model to train the data (effective for classification with small datasets)
text_clf = Pipeline([
     ('vect', CountVectorizer(ngram_range=(1,2))),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=0.0001, random_state = 40,
                           max_iter=5, tol=None))
])

# fitting the model to the training data
text_clf.fit(X_train, y_train)

# predicting the test labels using the trained model on the test features
predicted = text_clf.predict(X_test)

# printing the metrics
print(metrics.classification_report(y_test, predicted))

# converting the predictions (float) back to labels
labels_inv = { 0: 'Java Developer', 1: 'Software Engineer', 2: 'Programmer', 3: 'System Analyst', 4: 'Web Developer'}
predicted_labels = [labels_inv[number] for number in predicted]

# creating the dataset to upload
renamed_test_df = test_df.rename(columns={'Job_offer' : 'Job_description', 'Label' : 'Label_true'})
renamed_test_df['Label_pred'] = predicted_labels

# exporting the dataset as a csv file
renamed_test_df.to_csv('output_file.csv', sep=';')

# exporting the trained model
dump(text_clf, 'model.joblib')
