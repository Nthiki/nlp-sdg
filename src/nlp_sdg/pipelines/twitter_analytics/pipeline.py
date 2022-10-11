"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.twitter_analytics.nodes import  label_tweet


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [

            node(
                func=label_tweet,
                inputs="clean_data",
                outputs="clean_label_data",
                name="label_node",
            )
        
        ]
    )
    twitter_analytics = pipeline(
        pipe= pipeline_instance,
        inputs="clean_data",  
        outputs="clean_label_data",
        namespace = "twitter_analytics"
    )
    return twitter_analytics

