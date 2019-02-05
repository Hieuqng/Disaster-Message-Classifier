# General Purpose
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import warnings
import argparse

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
    '''
    Load data from sqlite database, quick cleaning, and
    return the features and labels of the models.

    INPUTS:
        database_filepath: path to the sqlite database

    OUTPUTS:
        X, Y: features and labels of the model
        Y.columns: categories of the label
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # 'related' column has values of 0,1,2 which doesn't make sense for binary classification
    df['related'] = df['related'].replace(2, 1)

    # 'child_alone' column has only value of 0. So our model will always predict 0.
    df = df.drop('child_alone', axis=1)

    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    '''
    Preprocess text features by tokenization and lemmatization.

    INPUTS:
        text: string of text need to be processed

    OUTPUTS:
        tokens: a list of tokens from the text
    '''

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def scorer(y_test, y_pred):
    '''
    Create a evaluation metric for the grid search.
    '''

    report = classification_report(y_test, y_pred, output_dict=True)
    weighted_avg = report['weighted avg']
    return weighted_avg['f1-score']


def build_model(pretrained_model):
    '''
    Build ML pipeline to including text processing and multi-output multi-class classifier
    If pretrained_model not None, load pretrained model. Otherwise output model is a grid search, which takes longer to train.
    '''

    if pretrained_model != None:
        clf = pickle.load(open(pretrained_model, 'rb'))
    else:
        pipeline = Pipeline([
            ('countvec', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(NbSvmClassifier()))
        ])

        parameters = {
            'clf__estimator__C': [1.0, 5.0, 10.0],
            'countvec__ngram_range': [(1, 1), (1, 2)]
        }

        f1_scorer = make_scorer(scorer)

        # optimize model
        clf = GridSearchCV(pipeline, parameters, scoring=f1_scorer,
                           cv=3, verbose=10)

    return clf


def evaluate_model(model, X_test, Y_test, category_names, pretrained_model):
    '''
    Evaluate the classifier by classification report (sklearn).
    If the pretrained_model is None, we trained the grid search, so best parameters will be reported.
    '''
    if pretrained_model == None:
        print("Best parameters: ", model.best_params_)

    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=category_names, output_dict=True)

    print("Validation Results: ")
    print(classification_report(Y_test, Y_pred, target_names=category_names))

    return (Y_pred, report)


def save_model(model, model_filepath):
    '''
    Save the model to a path specified by model_filepath.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    # Ignore warnings
    warnings.simplefilter('ignore')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Training Disaster Message Classifier')
    parser.add_argument('database_filepath', action='store', help='Directory of database file')
    parser.add_argument('model_filepath', action='store', help='Path to save the model')
    parser.add_argument('--pretrain', action='store', dest='pretrained_model',
                        help='Path to pretrained model')

    args = parser.parse_args()
    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    pretrained_model = args.pretrained_model

    # Train and Evaluate model
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model(pretrained_model)

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    Y_pred, report = evaluate_model(model, X_test, Y_test, category_names, pretrained_model)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
