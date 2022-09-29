"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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