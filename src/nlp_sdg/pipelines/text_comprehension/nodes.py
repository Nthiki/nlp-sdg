"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""
import numpy as np
import pandas as pd

from allennlp.predictors import Predictor
from allennlp_models.pretrained import load_predictor
def dummy_node(data):
    print("Text Comprehension dummy node completed")
    return 5

def locationOrganization(data):
        nlp_models = [
            { 'name' : 'ner-model',
            'url': 'C:/Users/Silas_Dell/Downloads/Compressed/ner-elmo.2021-02-12.tar.gz'
            },
        ]
        ## load models
        for nlp_model in nlp_models:       
            nlp_model['model'] = Predictor.from_path(nlp_model['url'])

    

        def entity_recognition (sentence):
            location = []
            for nlp_model in nlp_models:
                results =  nlp_model['model'].predict(sentence=sentence)
                for word, tag in zip(results["words"], results["tags"]):
                    if tag != 'U-LOC':
                        continue
                    else:
                        # print([word])#(f"{word}")
                        location.append(word)
                # print()
                return location

        def entity_recognition_pe(sentence):
            organisation = []
            for nlp_model in nlp_models:
                results =  nlp_model['model'].predict(sentence=sentence)
                for word, tag in zip(results["words"], results["tags"]):
                    if tag != 'U-ORG':
                        continue
                    else:
                        # print([word])#(f"{word}")
                        organisation.append(word)
                # print()
                return organisation
        result = []
        for i in range(len(data["text"])):
            result.append(list(set(entity_recognition(data["text"][i]))))
        re1 = []
        for i in range(len(data["text"])):
            re1.append(list(set(entity_recognition_pe(data["text"][i]))))
        data["location"]=result
        data["organisation"]=re1
        data['organisation'] = data.organisation.apply(lambda x: ', '.join([str(i) for i in x]))
        data['location'] = data.location.apply(lambda x: ', '.join([str(i) for i in x]))
        return data[["text","location","organisation"]]