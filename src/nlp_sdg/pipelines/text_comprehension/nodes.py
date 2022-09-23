"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""
pip install torch
pip install transformers
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')
import pandas as pd

def summarize_text(df, column, model, tokenizer):
    '''
    This function extract text from the text column of a pandas dataframe, model and tokenizer then it 
    produces a summary of that text and stores it in a new column in the dataframe
    '''
    result = []
    for i in range(len(df[column])):
        result.append(tokenizer.encode("summarize: "+df[column][i], return_tensors='pt', 
                                      max_length=tokenizer.model_max_length,
                                      truncation=True))
    
    df["tokens_input"] = result
    
    
    result_1 = []
    for i in range(len(df["tokens_input"])):
        result_1.append(model.generate(df["tokens_input"][i], min_length=80, 
                                      max_length=150, length_penalty=15, 
                                     early_stopping=True))
    
    df["summary_ids"] = result_1
    
    
    result_2 = []
    for i in range(len(df["summary_ids"])):
        result_2.append(tokenizer.decode((df["summary_ids"][i])[0], skip_special_tokens=True))
        
    
    df["summary"]=result_2
    df = df.drop(["tokens_input", "summary_ids"], axis=1)

    return df