"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from nlp_sdg.pipelines import data_engineering as de
from nlp_sdg.pipelines import twitter_analytics as ta
from nlp_sdg.pipelines import text_comprehension as tc
from nlp_sdg.pipelines import sdg_classification as sc
def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_engineering = de.create_pipeline()
    twitter_analytics = ta.create_pipeline()
    text_comprehension = tc.create_pipeline()
    sdg_classification = sc.create_pipeline()
    return {"__default__": data_engineering+twitter_analytics+text_comprehension+sdg_classification,
            "data_engineering": data_engineering,
            "twitter_analytics": twitter_analytics,
            "text_comprehension": text_comprehension,
            "sdg_classification": sdg_classification
            }
