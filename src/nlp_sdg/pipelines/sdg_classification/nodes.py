"""
This is a boilerplate pipeline 'sdg_classification'
generated using Kedro 0.18.2
"""
# utilities
from typing import List, Dict
import pandas as pd
import numpy as np
import logging

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV


''' ================================== 

     Data Science training

 ==================================== '''

def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    '''
    Splits data into training and test set.
    
     Args:
        data: Source processed data.
        parameters: Parameters defined in parameter.yml.
    
     Returns:
        A list containing split data.
        
    '''
    X = data['text'].values
    y = data['sdg'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], 
                                                        random_state=parameters['random_state'])
    return [X_train, X_test, y_train, y_test]



def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, parameters: Dict) -> List:
    '''
    Vectorize text column in train and test set.
    
     Args:
        X_train: Training text data.
        X_test: Testing text data.
        parameters: Parameters defined in parameter.yml.   
       
     Returns:
        A list containing vectorized train and test feature sets. 
        
    '''

    #E.g Tfid
    vectorizer = TfidfVectorizer(ngram_range=(parameters['ngram_range_min'],
                                              parameters['ngram_range_max']),
                                 max_features=parameters['max_features'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)


    return [X_train, X_test]


def train_model(X_train_vec: np.ndarray, y_train: np.ndarray, parameters: Dict) -> LinearSVC:
    '''
    Train the Linear SVC model.
    
     Args:
        X_train_vec: Vectorized training text data.
        y_train: Training data for SDG labels.
        parameters: Parameters defined in parameter.yml.
        
     Returns:
        Trained model.
        
    '''
    classifier = LinearSVC(max_iter=parameters['max_iter'], C=parameters['C'],
                           random_state=parameters['random_state'])
    #classifier = CalibratedClassifierCV(svc) 
    classifier.fit(X_train_vec, y_train)

    return classifier



def evaluate_model(sdg_classifier, X_test_vec: np.ndarray, y_test: np.ndarray):
    '''
    Generate and log classification report for test data.
    
     Args:
        X_test_vec: Vectorized test text data.
        y_test: Test data for SDG.
        classifier: Trained model.
        
    '''
    y_pred = sdg_classifier.predict(X_test_vec)
    #y_proba = sdg_classifier.predict_proba(X_test_vec)
    #best_n = np.argsort(y_proba, axis=1)[:,-1:]
    #best_1 = np.argmax(y_proba, axis=1)

    
    score = f1_score(y_test, y_pred, average='weighted')
    logger = logging.getLogger(__name__)
    logger.info("Model has an f1 score (weighted) of %.3f on test data.", score)
    #logger.info(classification_report(y_test, y_pred)) 

    
''' ================================== 

     ML predictions of new data

 ==================================== '''

#will add this node when Database is ready
#once we make predictions, we need to store these preds somewhere,
#so that they can be shown on streamlit
# we need to discuss, lets first keep a back log of data and predict it and show 
# on streamlit instead of having live updates and predictions
# new article data should be in pandas format with text col
# we should output a col with top 3 labels and their probabilites

def vectorize_new_text(training_data: pd.DataFrame, new_data: pd.DataFrame, parameters: Dict) -> List:
    '''
    Vectorize text column in new data coming from articles.
    
     Args:
        data: Full Training SDG text data.
        data: New article data
        parameters: Parameters defined in parameter.yml.   
       
     Returns:
        A list containing vectorized news article data. 
        
    '''

    #Original data
    X = training_data['text'].values
    y = training_data['sdg'].values
    #Data coming from the news articles
    X_news = new_data['text'].values

    #Tfid
    vectorizer = TfidfVectorizer(ngram_range=(parameters['ngram_range_min'],
                                              parameters['ngram_range_max']),
                                 max_features=parameters['max_features'])
    #Fit to ALL original train data and transform new data
    X = vectorizer.fit_transform(X)
    X_news = vectorizer.transform(X_news)


    return [X_news] #output should be X_news_vec


def get_predictions(sdg_classifier, X_news_vec: np.ndarray) -> List:
    '''
    Generate top 3 SDG predictions for incoming news articles
    
     Args:
        X_news_vec: Vectorized incoming news text data.
        classifier: Trained model.
        
    '''
    y_pred = sdg_classifier.predict(X_test_vec)

    return [y_pred]



