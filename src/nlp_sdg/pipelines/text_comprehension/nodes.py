"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""
import numpy as np
import pandas as pd

from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from tqdm import tqdm

def dummy_node(data):
    print("Text Comprehension dummy node completed")
    return 5

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