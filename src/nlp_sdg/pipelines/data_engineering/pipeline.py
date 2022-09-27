"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import dummy_node



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="rds_database",
                outputs="model_input_data",
                name="dummy_node",
            ),
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs="rds_database",
        namespace = "data_engineering",
        outputs = "model_input_data"
    )
    return data_engineering
