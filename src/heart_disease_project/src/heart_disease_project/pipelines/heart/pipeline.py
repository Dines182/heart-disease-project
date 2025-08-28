# src/<pkg>/pipelines/heart/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, build_preprocessor, train_rf, evaluate

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=dict(
                df="heart_raw",
                test_size="params:split.test_size",
                random_state="params:split.random_state",
            ),
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=build_preprocessor,
            inputs=[
                "params:numeric_features",
                "params:binary_features",
                "params:categorical_features",
            ],
            outputs="preprocessor",
            name="build_preprocessor_node",
        ),
        node(
            func=train_rf,
            inputs=dict(
                X_train="X_train",
                y_train="y_train",
                preprocessor="preprocessor",
                rf_params="params:rf",
            ),
            outputs=["model", "best_params"],
            name="train_model_node",
        ),
        node(
            func=evaluate,
            inputs=["model", "X_test", "y_test"],
            outputs="metrics",
            name="evaluate_model_node",
        ),
    ])
