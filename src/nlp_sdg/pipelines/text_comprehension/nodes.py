"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

import pandas as pd
from pyspark.sql import DataFrame
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz')


#questions in a list
questions = ["who is implicated",
              "which location is mentioned?"]

#column names for the respective questions
q_col_names = ['Q1', "Q2"]


def qes_and_ans(data: pd.DataFrame)-> pd.DataFrame:
        """
        QandA function to produce answers to pre-defined questions based on article text stored in a dataframe

        Args:
            data: Data containing a text column.
        Returns:
            data: A dataframe with new cols (Q1,Q2,,..) answering the questions based on the articles in each row 
        """
        #to remove this line in the future
        data = data.head(5)

        for question, q_col_name in zip(questions, q_col_names):
            for i in range(len(data["text"])):
                data[q_col_name] = predictor.predict(passage=data["text"][i], question=question)["best_span_str"]

        return data 
