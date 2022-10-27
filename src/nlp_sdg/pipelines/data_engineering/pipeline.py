"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline


from nlp_sdg.pipelines.data_engineering.nodes import osdg_preprocessed_data, preprocess_tweets
from nlp_sdg.pipelines.data_engineering.nodes import article_preprocessed_data, fetch_all_tweets, twitter_rds_to_csv


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=osdg_preprocessed_data,
                inputs="sdg_data",
                outputs="osdg_preprocessed_data",
                name="osdg_preprocess_data_node",
            ),
            node(
                func=article_preprocessed_data,
                inputs="articles",
                outputs="cleaned_articles",
                name="articles_preprocess_data_node",
            ),
            #   node(
            #     func=data_preprocessing,
            #     inputs="twitter_data",
            #     outputs="cleaned_twitter_data",
            #     name="create_twitter_input_table_node",
            # ),

            #node(
            #     func=fetch_all_tweets,
            #     inputs=None,
            #     outputs='save_data_to_rds',
            #     name= "fetch_save_tweets_node",
            # ),

            node(
                func= twitter_rds_to_csv,
                inputs='raw_tweet_data',
                outputs='clean_tweet_data_s3',
                name='twitter_rds_to_csv_node'
            ),
            #node(
            #    func= preprocess_tweets,
                # inputs='tweet_text_data',
            #    inputs='raw_tweet_data',
            #    outputs='clean_tweet_data',
            #    name='preprocess_tweets_node'
            #)
        ]
    )
    data_engineering = pipeline(
        pipe=pipeline_instance,
        inputs=["sdg_data","articles","raw_tweet_data"],
        namespace = "data_engineering",
        outputs = ["osdg_preprocessed_data","cleaned_articles","clean_tweet_data_s3"]
    )
    return data_engineering
