"""
This is a boilerplate pipeline 'sdg_classification'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.sdg_classification.nodes import split_data, vectorize_text, train_model, evaluate_model


def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["sdg_model_input_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=vectorize_text,
                inputs=["X_train", "X_test", 'parameters'],
                outputs=['X_train_vec', 'X_test_vec'],
                name="vectorize_text_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_vec", "y_train","parameters"],
                outputs="ml_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["ml_model", "X_test_vec", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            )
        ]
    )

