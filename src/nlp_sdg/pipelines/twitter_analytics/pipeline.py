"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.twitter_analytics.nodes import dummy_node, label_tweet


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [

            node(
                func=label_tweet,
                inputs="clean_tweet_data",
                outputs="clean_label_data",
                name="label_node",
            )
        
        ]
    )
    twitter_analytics = pipeline(
        pipe= pipeline_instance,
        inputs=["model_input_data", "clean_tweet_data"],  
        outputs=["analytics_output", "clean_label_data"],
        namespace = "twitter_analytics"
    )
    return twitter_analytics

