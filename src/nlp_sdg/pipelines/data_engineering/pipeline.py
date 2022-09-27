"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import clean_agreement, preprocess_data, data_balancing



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=clean_agreement,
                inputs="sdg_data",
                outputs="filtered_data",
                name="clean_agreement_node",
            ),
            node(
                func=preprocess_data,
                inputs="filtered_data",
                outputs="cleaned_data",
                name="preprocess_data_node",
            ),
            node(
                func=data_balancing,
                inputs="cleaned_data",
                outputs="model_input_data",
                name="data_balancing_node",
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
