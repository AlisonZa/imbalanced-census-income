from scipy.stats import skew
from typing import Dict
from src.entities import FeatureDefinition
import pandas as pd

def calculate_skewness(feature_definition: FeatureDefinition) -> pd.DataFrame:
    """
    Calculate the skewness of each numeric feature in the dataset.
    
    Args:
        feature_definition (FeatureDefinition): An object containing dataset details.
    
    Returns:
        pd.DataFrame: A DataFrame with feature names and their corresponding skewness values.
    """
    df = feature_definition.data_frame
    numeric_features = feature_definition.numeric
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("data_frame must be a pandas DataFrame")
    
    skewness_values = {feature: skew(df[feature].dropna()) for feature in numeric_features}
    
    return pd.DataFrame(list(skewness_values.items()), columns=["Feature", "Skewness"])
