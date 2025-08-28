# src/<pkg>/pipeline_registry.py
from __future__ import annotations
from kedro.pipeline import Pipeline
from heart_disease_project.pipelines.heart.pipeline import create_pipeline as heart_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "heart": heart_pipeline(),
        "__default__": heart_pipeline(),
    }
