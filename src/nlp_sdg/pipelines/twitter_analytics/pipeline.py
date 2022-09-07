"""
This is a boilerplate pipeline 'twitter_analytics'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.twitter_analytics.nodes import dummy_node


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="data_engineering.model_input_data",
                outputs="analytics_output",
                name="dummy_node",
            ),
        ]
    )
    twitter_analytics = pipeline(
        pipe=pipeline_instance,
        inputs="data_engineering.model_input_data",
        namespace = "twitter_analytics"
    )
    return twitter_analytics
