"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.data_engineering.nodes import dummy_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dummy_node,
                inputs="sdg_data",
                outputs="dummy_data",
                name="dummy_node"
            )
        ]
    )
