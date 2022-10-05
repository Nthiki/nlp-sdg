"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.text_comprehension.nodes import qes_and_ans



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=qes_and_ans,
                inputs="sdg_data",
                outputs="q_and_a_data",
                name="q_and_a_node",
            ),
        ]
    )
    text_comprehension = pipeline(
        pipe=pipeline_instance,
        inputs="sdg_data",
        namespace = "text_comprehension",
        outputs = "q_and_a_data"
    )
    return text_comprehension