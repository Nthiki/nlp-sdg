"""
This is a boilerplate pipeline 'data_processing_classification'
generated using Kedro 0.18.2
"""


from kedro.pipeline import Pipeline, node, pipeline
from nlp_sdg.pipelines.data_processing_classification.nodes import clean_agreement, preprocess_data

def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                func = clean_agreement,
                inputs='sdg_text_data',
                outputs='cleaned_agreement_data',
                name="clean_agreemet_node",
            ),
            node(
                func = preprocess_data,
                inputs='cleaned_agreement_data',
                outputs='sdg_model_input_table',
                name="create_sdg_model_input_table_node",
            )
        ]
    ) 
