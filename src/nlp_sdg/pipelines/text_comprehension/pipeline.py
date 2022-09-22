"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.text_comprehension.nodes import dummy_node,locationOrganization



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=dummy_node,
                inputs="model_input_data",
                outputs="comprehension_output",
                name="dummy_node",
            ),
            node(
                func=locationOrganization,
                inputs="sdg_text_data",
                outputs="locations_and_org",
                name="locationOrganization_node",
            ),
        ]
    )
    text_comprehension = pipeline(
        pipe=pipeline_instance,
        inputs=["model_input_data","sdg_text_data"]
        namespace = "text_comprehension"
        outputs ="locations_and_org"
    )
    return text_comprehension