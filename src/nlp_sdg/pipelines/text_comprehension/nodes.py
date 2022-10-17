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




predictor = Predictor.from_path('references/bidaf-elmo-model-2020.03.19.tar.gz')


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

nlp_model = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz')

def get_organization(data):
    def get_org(df_org):
        df_ent = []
        for _, row in pd.DataFrame({"beg": df_org.loc[lambda x: x["tags"] == "B-ORG"].index.values,
                    "end": df_org.loc[lambda x: x["tags"] == "L-ORG"].index.values + 1}).iterrows():
            df_ent.append(df_org.iloc[row["beg"]:row["end"]]["words"].str.cat(sep=" "))
        df_ent.extend(df_org.loc[lambda x: x["tags"] == "U-ORG"]["words"].to_list())
        return df_ent


    df_edges = []
    for _, row in tqdm(list(data.iterrows())):
        df_org = []
        sents = SpacySentenceSplitter().split_sentences(row["text"])
        for i, s in list(enumerate(sents)):
            res = nlp_model.predict(
            sentence=s
            )
            df_org.append(pd.DataFrame({"tags": res["tags"], "words": res["words"], "text": row["text"]})\
            .loc[lambda x: x["tags"].str.contains("ORG")])

        df_org = pd.concat(df_org).reset_index(drop=True)
        df_ent = get_org(df_org)
        df_edges.append(pd.DataFrame({"text": row["text"], "organization": df_ent}))

    df_org = pd.concat(df_edges)
    df_org = pd.DataFrame(df_org.groupby(df_org["text"])["organization"].apply(lambda x: ', '.join(np.unique(x.values.ravel()))).reset_index())

    return df_org["organization"]


def get_location(data):
    def get_location(df_loc):
        df_ent = []
        for _, row in pd.DataFrame({"beg": df_loc.loc[lambda x: x["tags"] == "B-LOC"].index.values,
                        "end": df_loc.loc[lambda x: x["tags"] == "L-LOC"].index.values + 1}).iterrows():
            df_ent.append(df_loc.iloc[row["beg"]:row["end"]]["words"].str.cat(sep=" "))
        df_ent.extend(df_loc.loc[lambda x: x["tags"] == "U-LOC"]["words"].to_list())
        return df_ent


    df_edges = []
    for _, row in tqdm(list(data.iterrows())):
        df_loc = []
        sents = SpacySentenceSplitter().split_sentences(row["text"])
        for i, s in list(enumerate(sents)):
            res = nlp_model.predict(
            sentence=s
            )
            df_loc.append(pd.DataFrame({"tags": res["tags"], "words": res["words"], "text": row["text"]})\
            .loc[lambda x: x["tags"].str.contains("LOC")])

        df_loc = pd.concat(df_loc).reset_index(drop=True)
        df_ent = get_location(df_loc)
        df_edges.append(pd.DataFrame({"text": row["text"], "location": df_ent}))


    df_loc = pd.concat(df_edges)
    df_loc = pd.DataFrame(df_loc.groupby(df_loc["text"])["location"].apply(lambda x: ', '.join(np.unique(x.values.ravel()))).reset_index())

    return df_loc["location"]
