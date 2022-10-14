"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline


from nlp_sdg.pipelines.data_engineering.nodes import osdg_preprocessed_data, preprocess_tweets


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=osdg_preprocessed_data,
                inputs="sdg_data",
                outputs="osdg_preprocessed_data",
                name="osdg_preprocess_data_node",
            ),

            node(
                func= preprocess_tweets,
                # inputs='tweet_text_data',
                inputs='raw_tweet_data',
                outputs='clean_tweet_data',
                name='preprocess_tweets_node'
            )

        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs=["sdg_data","raw_tweet_data"],
        namespace = "data_engineering",
        outputs = ["osdg_preprocessed_data", "clean_tweet_data"]
    )
    return data_engineering
