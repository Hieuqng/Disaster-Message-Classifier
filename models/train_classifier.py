# General Purpose
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import warnings

# Text Processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from nbsvm import NbSvmClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    df['related'] = df['related'].replace(2, 1)
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    Y.drop('child_alone', axis=1, inplace=True)
    return X, Y, Y.columns


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def scorer(y_test, y_pred, metric):
    report = classification_report(y_test, y_pred, output_dict=True)
    weighted_avg = report['weighted avg']
    return weighted_avg[metric]


def build_model(model_filepath=None):
    '''
    Build ML pipeline to including text processing and multi-output multi-class classifier
    If tuning=True, output model is a grid search, which takes longer to train.
    '''

    if model_filepath != None:
        clf = pickle.load(open(model_filepath, 'rb'))

    else:
        clf = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(NbSvmClassifier(C=10.0))),
        ])

    return clf


def evaluate_model(model, X_test, Y_test, category_names, verbose=False):
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=category_names, output_dict=True)

    if verbose:
        print(classification_report(Y_test, Y_pred, target_names=category_names))

    return (Y_pred, report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) <= 4:
        # Ignore warnings
        warnings.simplefilter('ignore')

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        Y_pred, report = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db model_name.pkl')


if __name__ == '__main__':
    main()
