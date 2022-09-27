"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import dummy_node, clean_agreement



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="sdg_data",
                outputs="model_input_data",
                name="dummy_node",
            ),
            node(
                func=clean_agreement,
                inputs="sdg_text_data", 
                outputs="cleaned_agreement_data",
                name="clean_agreement_node"
            ),
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs=["sdg_data", "sdg_text_data"],
        namespace = "data_engineering",
        outputs = ["model_input_data", "cleaned_agreement_data"]
    )
    return data_engineering
    
