import os
import pandas as pd
import numpy as np
from typing import List, Optional
from src.entities import FeatureDefinition, EnvironmentConfiguration
from itertools import combinations


def generate_all_two_way_tables(feature_def: FeatureDefinition, config: EnvironmentConfiguration) -> None:
    """
    Generates two-way tables for all combinations of categorical features and saves them to Excel.
    
    Args:
        feature_def: FeatureDefinition object containing the dataset and feature information
        config: EnvironmentConfiguration object with folder paths
    """
    # Collect all categorical features
    categorical_features = []
    if feature_def.categorical_ordinals:
        categorical_features.extend(feature_def.categorical_ordinals)
    if feature_def.categorical_nominals:
        categorical_features.extend(feature_def.categorical_nominals)
    if feature_def.categorical_binary:
        categorical_features.extend(feature_def.categorical_binary)
    
    # Create output directory if it doesn't exist
    os.makedirs(config.two_way_tables_folder, exist_ok=True)
    
    # Generate tables for all combinations of categorical features
    for feature1, feature2 in combinations(categorical_features, 2):
        # Create contingency table
        cont_table = pd.crosstab(
            feature_def.data_frame[feature1],
            feature_def.data_frame[feature2],
            margins=True
        )
        
        # Get unique values (excluding 'All')
        feature2_values = list(cont_table.columns[:-1])
        
        # Initialize the final table structure
        rows = []
        
        # Create column names with arrows
        feature1_name = f"{feature1} ↓"
        feature2_name = f"{feature2} →"
        
        # Process each row except the last one (Total)
        for idx in cont_table.index[:-1]:
            # Row percentages
            row_pcts = (cont_table.loc[idx, :] / cont_table.loc[idx, 'All'] * 100).round(1)
            row_data = {
                feature1_name: idx,
                feature2_name: 'Row %'
            }
            row_data.update({str(val): row_pcts[val] for val in feature2_values})
            row_data['Sum Row'] = 100
            rows.append(row_data)
            
            # Column percentages
            col_pcts = (cont_table.loc[idx, :] / cont_table.loc['All', :] * 100).round(1)
            row_data = {
                feature1_name: idx,
                feature2_name: 'Column %'
            }
            row_data.update({str(val): col_pcts[val] for val in feature2_values})
            row_data['Sum Row'] = '-'
            rows.append(row_data)
            
            # Total percentages
            total_pcts = (cont_table.loc[idx, :] / cont_table.values[-1, -1] * 100).round(1)
            row_data = {
                feature1_name: idx,
                feature2_name: 'Total %'
            }
            row_data.update({str(val): total_pcts[val] for val in feature2_values})
            row_data['Sum Row'] = total_pcts[:-1].sum()
            rows.append(row_data)
        
        # Add Column Total rows
        # Row percentages for Column Total (should be '-' as it's not applicable)
        row_data = {
            feature1_name: 'Column Total',
            feature2_name: 'Row %'
        }
        row_data.update({str(val): '-' for val in feature2_values})
        row_data['Sum Row'] = '-'
        rows.append(row_data)
        
        # Column percentages for Column Total (should be 100% for each column)
        row_data = {
            feature1_name: 'Column Total',
            feature2_name: 'Column %'
        }
        row_data.update({str(val): 100.0 for val in feature2_values})
        row_data['Sum Row'] = 100
        rows.append(row_data)
        
        # Total percentages for Column Total (actual percentage of total for each column)
        total_sum = cont_table.loc['All', 'All']
        col_totals = cont_table.loc['All', feature2_values]
        total_col_pcts = (col_totals / total_sum * 100).round(1)
        row_data = {
            feature1_name: 'Column Total',
            feature2_name: 'Total %'
        }
        row_data.update({str(val): pct for val, pct in zip(feature2_values, total_col_pcts)})
        row_data['Sum Row'] = 100
        rows.append(row_data)
        
        # Create and save DataFrame
        result_df = pd.DataFrame(rows)
        output_path = os.path.join(config.two_way_tables_folder, f"two_way_table_{feature1}_{feature2}.xlsx")
        result_df.to_excel(output_path, index=False)
