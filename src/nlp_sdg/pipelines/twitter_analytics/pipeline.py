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
                func=dummy_node,
                inputs="model_input_data",
                outputs="analytics_output",
                name="dummy_node",
            ),
            node(
                func=label_tweet,
                inputs="cleaned_twitter_data",
                outputs="labelled_twitter_data",
                name="label_twitter_node",
            ),
        ]
    )
    twitter_analytics = pipeline(
        pipe=pipeline_instance,
        inputs=["model_input_data","cleaned_twitter_data"],
        outputs = ["analytics_output", "labelled_twitter_data"],
        namespace = "twitter_analytics"
    )
    return twitter_analytics
