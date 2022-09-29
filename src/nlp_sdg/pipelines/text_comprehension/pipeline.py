"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.text_comprehension.nodes import  summarize_text



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=summarize_text,
                inputs="rds_articles",
                outputs="summarized_text_data",
                name="summarize_text_node",
            ),
        ]
    )
    text_comprehension = pipeline(
        pipe=pipeline_instance,
        inputs="rds_articles",
        namespace = "text_comprehension"
    )
    return text_comprehension