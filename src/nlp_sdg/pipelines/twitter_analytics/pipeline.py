"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.twitter_analytics.nodes import label_tweet


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
        
            node(
                func=label_tweet,
                inputs="clean_tweet_data_s3",
                outputs="labelled_twitter_data",
                name="label_twitter_node",
            ),
        ]
    )
    twitter_analytics = pipeline(
        pipe=pipeline_instance,
        inputs="clean_tweet_data_s3",
        outputs = "labelled_twitter_data",
        namespace = "twitter_analytics"
    )
    return twitter_analytics

