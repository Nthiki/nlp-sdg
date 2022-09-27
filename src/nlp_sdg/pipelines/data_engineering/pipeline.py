"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import clean_agreement, clean_text, data_balancing



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=clean_agreement,
                inputs="sdg_text_data", 
                outputs="cleaned_agreement_data",
                name="clean_agreement_node"
            ),
            node(
                func=clean_text,
                inputs="cleaned_agreement_data", 
                outputs="cleaned_text_data",
                name="clean_text_node"
            ),
            node(
                func=data_balancing,
                inputs="cleaned_text_data", 
                outputs="model_input_data",
                name="data_balancing_node"
            ),
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs=["sdg_text_data"],
        namespace = "data_engineering",
        outputs = ["cleaned_agreement_data", "cleaned_text_data", "model_input_data"]
    )
    return data_engineering
    
