"""
This is a boilerplate pipeline 'sdg_classification'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.sdg_classification.nodes import dummy_node


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="model_input_data",
                outputs="classification_output",
                name="dummy_node_classification",
            ),
        ]
    )
    text_classification = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_data",
        namespace = "text_classification"
    )
    return text_classification
