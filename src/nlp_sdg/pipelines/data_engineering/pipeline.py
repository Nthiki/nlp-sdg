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
                func=convert_to_csv,
                inputs="test_data",
                outputs= "confirm_data",
                name= "confirm_data_node",
            )
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs= "test_data",
        namespace = "data_engineering",
        outputs = "confirm_data"
    )
    return data_engineering
