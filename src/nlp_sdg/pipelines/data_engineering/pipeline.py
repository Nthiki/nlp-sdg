"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import  data_preprocessing



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
          
            node(
                func = data_preprocessing,
                inputs = 'max_date',
                outputs = "save_data_to_rds",
                name = 'data_preprocessing_node'
            ),

        ]
    )
    data_engineering = pipeline(
        pipe = pipeline_instance,
        outputs = "save_data_to_rds",
        inputs = "max_date",
        namespace = "data_engineering"        
    )
    return data_engineering
