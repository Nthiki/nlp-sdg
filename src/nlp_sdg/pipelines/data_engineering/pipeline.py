"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import  data_preprocessing,fetch_save_tweets



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=fetch_save_tweets,
                inputs=None,
                outputs= 'save_data_to_rds',
                name= "fetch_save_tweets_node",
            ),
          
            node(
                func= data_preprocessing,
                # inputs='tweet_text_data',
                inputs='raw_data',
                outputs='clean_tweet_data',
                name='data_preprocessing_node'
            )
        ]
    )
    data_engineering = pipeline(
        pipe= pipeline_instance,
        outputs = ["save_data_to_rds","clean_tweet_data"],
        inputs="raw_data",
        namespace = "data_engineering"        
    )
    return data_engineering
