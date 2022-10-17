"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.text_comprehension.nodes import qes_and_ans, get_location, get_organization



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=qes_and_ans,
                inputs="articles",
                outputs="q_and_a_data",
                name="q_and_a_node",
            ),
            node(
                func=get_organization,
                inputs="articles",
                outputs="organization_data",
                name="get_organization_node",
            ),
            node(
                func=get_location,
                inputs="articles",
                outputs="location_data",
                name="get_location_node"
            )
        ]
    )
    text_comprehension = pipeline(
        pipe=pipeline_instance,
        inputs="articles",
        namespace = "text_comprehension",
        outputs = ["q_and_a_data","location_data","organization_data"]
    )
    return text_comprehension