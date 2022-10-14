"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

import pandas as pd
from pyspark.sql import DataFrame
from allennlp.predictors.predictor import Predictor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "references/t5-base/"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def summarize_text(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe.
    Then it returns a dataframe that creates summarized column of the text column.
    Args:
        source training data
    Returns:
        data with a "text" containing the summarized text
    '''
    data = data.head(3)
    result = []
    for i in range(len(data["text"])):
        result.append(tokenizer.encode("summarize: "+data["text"][i], return_tensors='pt', 
                                      max_length=tokenizer.model_max_length,
                                      truncation=True))

    data["tokens_input"] = result


    result_1 = []
    for i in range(len(data["tokens_input"])):
        result_1.append(model.generate(data["tokens_input"][i], min_length=80, 
                                      max_length=150, length_penalty=15, 
                                     early_stopping=True))

    data["summary_ids"] = result_1


    result_2 = []
    for i in range(len(data["summary_ids"])):
        result_2.append(tokenizer.decode((data["summary_ids"][i])[0], skip_special_tokens=True))


    data["summary"]=result_2

    data = data.drop(["tokens_input", "summary_ids"], axis=1)

    return data




predictor = Predictor.from_path('~/references/bidaf-elmo-model-2020.03.19.tar.gz')


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
