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

def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    '''
    Splits data into training and test set.
    
     Args:
        data: Source processed data.
        parameters: Parameters defined in parameter.yml.
    
     Returns:
        A list containing split data.
        
    '''
    #data = _handle_empty(data) ... what happens if there is missing data? We could create a function for this as well
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

    #vectorizer = CountVectorizer()
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
    classifier.fit(X_train_vec, y_train)
    return classifier


def evaluate_model(ml_model, X_test_vec: np.ndarray, y_test: np.ndarray):
    '''
    Generate and log classification report for test data.
    
     Args:
        X_test_vec: Vectorized test text data.
        y_test: Test data for SDG.
        classifier: Trained model.
        
    '''
    y_pred = ml_model.predict(X_test_vec)
    
    score = f1_score(y_test, y_pred, average='weighted')
    logger = logging.getLogger(__name__)
    logger.info("Model has an f1 score (weighted) of %.3f on test data.", score)
    #logger.info(classification_report(y_test, y_pred)) 
    
    #in kedro we will not return score, only log what we have done
    #return score