"""
This is a boilerplate pipeline 'sdg_classification'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.sdg_classification.nodes import split_data, vectorize_text, train_model
from nlp_sdg.pipelines.sdg_classification.nodes import evaluate_model,vectorize_new_text, get_predictions


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=split_data,
                inputs=["osdg_preprocessed_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=vectorize_text,
                inputs=["X_train", "X_test","parameters"],
                outputs=['X_train_vec', 'X_test_vec', "vectorizer"],
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
            node(
                func=vectorize_new_text,
                inputs=["cleaned_articles", "vectorizer"],
                outputs='X_news_vec',
                name="vectorize_new_text_node",
            ),
            node(
                func=get_predictions,
                inputs=['sdg_classifier', 'X_news_vec'],
                outputs='predictions',
                name="predict_new_text_node",
            ),
        ]
    )
    text_classification = pipeline(
        pipe=pipeline_instance,
        inputs=["osdg_preprocessed_data","cleaned_articles"],
        namespace = "text_classification",
        outputs = 'predictions'
    )
    return text_classification
