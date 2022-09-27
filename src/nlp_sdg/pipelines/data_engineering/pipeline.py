"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import dummy_node, clean_agreement, preprocess_sdg_data



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=osdg_preprocessed_data,
                inputs="sdg_data",
                outputs="osdg_preprocessed_data",
                name="osdg_preprocess_data_node",
            ),
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs= "sdg_data",
        namespace = "data_engineering",
        outputs = "osdg_preprocessed_data",
    )
    return data_engineering
