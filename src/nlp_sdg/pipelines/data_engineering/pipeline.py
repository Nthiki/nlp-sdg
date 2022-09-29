"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import convert_to_csv, dummy_node



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="raw_data",
                outputs= "articles_data",
                name= "articles_data_node",
            ),
            node(
                func=dummy_node,
                inputs="S3_bucket",
                outputs= "S3_data",
                name= "s3_data_node",
            )
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs= ["raw_data","S3_bucket"],
        namespace = "data_engineering",
        outputs = ["articles_data","S3_data"]
    )
    return data_engineering
