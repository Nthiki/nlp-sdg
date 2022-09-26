"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.twitter_analytics.nodes import dummy_node, data_preprocessing


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="model_input_data",
                outputs="analytics_output",
                name="dummy_node",
            ),
            node(
                func= data_preprocessing,
                inputs='tweet_text_data',
                outputs='clean_tweet_data',
                name='data_preprocessing_node'
            )
        ]
    )
    twitter_analytics = pipeline(
        pipe=pipeline_instance,
        inputs=["model_input_data", "tweet_text_data"],
        outputs=["clean_tweet_data", "analytics_output"],
        namespace = "twitter_analytics"
    )
    return twitter_analytics
