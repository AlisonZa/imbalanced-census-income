from pydantic import BaseModel, Field, field_validator
from typing import Any

from typing import List, Optional
import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class FeatureDefinition():
    """
    Definition of dataset features including target, categorical, numerical, and task type.

    Attributes:
        data_frame (pd.DataFrame): The dataset as a Pandas DataFrame.
        type_of_task (str): The type of task, one of "regression", "clustering", "binary_classification", "multiclass_classification".
        categorical_ordinals (Optional[List[str]]): List of ordinal categorical features.
        categorical_nominals (Optional[List[str]]): List of nominal categorical features.
        categorical_binary (Optional[List[str]]): List of binary categorical features.
        time_stamps (Optional[List[str]]): List of time-related features.
        numeric (Optional[List[str]]): List of numerical features.
    """

    data_frame: Any
    type_of_task: str

    categorical_ordinals: Optional[List[str]] = None
    categorical_nominals: Optional[List[str]] = None
    categorical_binary: Optional[List[str]] = None
    time_stamps: Optional[List[str]] = None
    numeric: Optional[List[str]] = None


class EnvironmentConfiguration(BaseModel):
    artifacts_folder: str = os.path.join("artifacts")

    # Data
    raw_data_folder: str = os.path.join("raw_data", "CC GENERAL.csv")

    # Exploratory Data Analysys
    eda_folder: str = os.path.join(artifacts_folder, "exploratory_data_analysys")
    two_way_tables_folder: str = os.path.join(eda_folder, "two_way_tables")
    y_data_profiling_folder: str = os.path.join(eda_folder, "y_data_profiling")
    y_data_profiling_file: str = os.path.join(eda_folder, "y_data_profiling", "data_profiling.html")

    unique_values_spreadsheet: str = os.path.join(eda_folder, "unique_values_spreadsheet.xlsx")

    # Plots
    plots_folder: str = os.path.join(eda_folder, "plots")
    univariate_plots_folder: str = os.path.join(plots_folder, "univariate")
    bivariate_plots_folder: str = os.path.join(plots_folder, "bivariate")

    # Pipelines
    pipelines_folder: str = os.path.join(artifacts_folder, "pipelines")






