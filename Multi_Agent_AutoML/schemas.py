from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CleaningAction(BaseModel):
    action: str               # "impute_missing" | "drop_column"
    column: str
    rationale: str

class DataCleanerReport(BaseModel):
    inspected_columns: List[str]
    actions_taken: List[CleaningAction]
    dropped_columns: List[str]
    imputed_columns: List[str]
    final_row_count: int
    final_column_count: int
    summary: str


class FeatureEngineeringAction(BaseModel):
    action: str              # encode_categorical | create_interaction | select_top_features
    details: str
    rationale: str

class FeatureEngineerReport(BaseModel):
    target_column: str
    encoded_columns: List[str]
    created_features: List[str]
    selected_feature_count: int
    actions_taken: List[FeatureEngineeringAction]
    summary: str

