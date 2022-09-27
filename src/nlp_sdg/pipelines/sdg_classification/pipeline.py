"""
This is a boilerplate pipeline 'sdg_classification'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.sdg_classification.nodes import dummy_node, split_data, vectorize_text, train_model
from nlp_sdg.pipelines.sdg_classification.nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="model_input_data",
                outputs="classification_output",
                name="dummy_node_classification",
            ),
            node(
                func=split_data,
                inputs=["sdg_model_input_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=vectorize_text,
                inputs=["X_train", "X_test","parameters"],
                outputs=['X_train_vec', 'X_test_vec'],
                name="vectorize_text_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_vec", "y_train", "parameters"],
                outputs="sdg_classifier",
                name="train_sdg_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["sdg_classifier", "X_test_vec", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
    text_classification = pipeline(
        pipe=pipeline_instance,
        inputs=["model_input_data","sdg_model_input_table"],
        namespace = "text_classification"
    )
    return text_classification
