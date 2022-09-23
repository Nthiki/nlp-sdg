"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""
import numpy as np
import pandas as pd
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz')

def dummy_node(data):
    print("Text Comprehension dummy node completed")
    return 5

def qes_and_ans(data,questions = "Is Shell implicated?"):

        def qestions(data,question):
            result = []
            for i in range(len(data["text"])):
                result.append(predictor.predict(passage=data["text"][i], question=question)["best_span_str"])
            return result

        data[questions] = qestions(data, question = questions)
        return data[questions]