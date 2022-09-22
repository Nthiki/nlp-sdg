"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import preprocess_data



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=preprocessed_data,
                inputs="sdg_data",
                outputs="preprocessed_data",
                name="preprocessed_data",
            ),
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs="sdg_data",
        namespace = "data_engineering",
        outputs = "model_input_data"
    )
    return data_engineering
