from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from missforest import MissForest
import pandas as pd
from featuretools.selection import remove_highly_correlated_features


def standard_scale_dataframe(df):
    """
    Standardizes the numerical columns in the given dataframe using StandardScaler.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        scaled_df (pd.DataFrame): The transformed dataframe with standardized numerical columns.
        scaler (StandardScaler): The trained StandardScaler object.
    """
    # Select only numeric columns for scaling
    num_columns = df.select_dtypes(include=['int64', 'float']).columns
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Apply scaling to numeric columns
    df_scaled = df.copy()
    df_scaled[num_columns] = scaler.fit_transform(df[num_columns])
    
    return df_scaled, scaler

def minmax_scale_dataframe(df):
    """
    Scales the numerical columns in the given dataframe using MinMaxScaler.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        scaled_df (pd.DataFrame): The transformed dataframe with MinMax scaled numerical columns.
        scaler (MinMaxScaler): The trained MinMaxScaler object.
    """
    # Select only numeric columns for scaling
    num_columns = df.select_dtypes(include=['int64', 'float']).columns
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Apply scaling to numeric columns
    df_scaled = df.copy()
    df_scaled[num_columns] = scaler.fit_transform(df[num_columns])
    
    return df_scaled, scaler

def robust_scale_dataframe(df):
    """
    Scales the numerical columns in the given dataframe using RobustScaler.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        scaled_df (pd.DataFrame): The transformed dataframe with Robust scaled numerical columns.
        scaler (RobustScaler): The trained RobustScaler object.
    """
    # Select only numeric columns for scaling
    num_columns = df.select_dtypes(include=['int64', 'float']).columns
    
    # Initialize the RobustScaler
    scaler = RobustScaler()
    
    # Apply scaling to numeric columns
    df_scaled = df.copy()
    df_scaled[num_columns] = scaler.fit_transform(df[num_columns])
    
    return df_scaled, scaler

def equal_width_binning(df, columns, n_bins, return_full_dataset=True):
    """
    Apply equal-width binning to specified columns of a pandas DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing the data to bin.
    columns : list of str
        List of column names to apply equal-width binning.
    n_bins : int
        The number of bins to create for each specified column.
    return_full_dataset : bool, optional, default=True
        If True, return the entire dataset with the binned columns added.
        If False, return only the binned columns.
        
    Returns:
    -------
    pd.DataFrame
        A new dataframe with equal-width binned columns.
        If `return_full_dataset` is False, only the binned columns are returned.
    """
    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    
    for col in columns:
        if col in df_copy.columns:
            # Apply equal-width binning and store the result as a new column
            binned_col_name = f"{col}_binned"
            df_copy[binned_col_name] = pd.cut(df_copy[col], bins=n_bins, labels=False)
        else:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
    
    if return_full_dataset:
        return df_copy
    else:
        # Return only the newly binned columns
        binned_columns = [f"{col}_binned" for col in columns]
        return df_copy[binned_columns]

def missforest_imputation(df):
    """
    Impute missing values in a pandas DataFrame using the MissForest algorithm.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing missing values to be imputed.
        
    Returns:
    -------
    tuple
        A tuple containing:
        - pd.DataFrame: The dataframe with missing values imputed.
        - MissForest: The trained MissForest imputer object.
    
    Notes:
    -----
    MissForest is a Random Forest-based imputation method that handles both
    numeric and categorical data. This function works with mixed data types.
    """
    df_copy = df.copy()  # Create a copy to avoid modifying the original data
    
    # Initialize the MissForest imputer
    imputer = MissForest()

    # Fit the imputer and impute missing values
    imputed_array = imputer.fit_transform(df_copy)
    
    # Convert the imputed array back to a pandas DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=df_copy.columns, index=df_copy.index)
    
    return imputed_df, imputer

def drop_highly_correlated_features(df, threshold=0.85):
    """
    Remove highly correlated features from a pandas DataFrame using Featuretools.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing the features.
    threshold : float, optional (default=0.85)
        Correlation threshold above which features are considered highly correlated
        and removed.
    
    Returns:
    -------
    pd.DataFrame
        A new dataframe with highly correlated features removed.
    """
    df_copy = df.copy()  # Avoid modifying the original DataFrame
    
    # Remove highly correlated features
    reduced_df = remove_highly_correlated_features(df_copy, pct_corr_threshold=threshold)
    
    return reduced_df

def apply_transformations_v2(df, columns=None, epsilon=1e-10):
    """
    Apply various log and power transformations to specified columns in a dataframe.
    Returns a single dataframe with all transformations as new columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to transform. If None, transforms all numeric columns
    epsilon : float, optional
        Small constant to add to handle zeros and negative values
        
    Returns:
    --------
    pandas.DataFrame : Original dataframe with additional columns for each transformation
    """
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = result_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Apply transformations
    for col in columns:
        # Store column values for reuse
        col_values = result_df[col].values
        abs_values = np.abs(col_values)
        signs = np.sign(col_values)
        
        # Log transformations
        # 1. Natural log transformation (ln(x + epsilon))
        result_df[f'{col}_ln'] = np.log(abs_values + epsilon)
        
        # 2. Log10 transformation
        result_df[f'{col}_log10'] = np.log10(abs_values + epsilon)
        
        # 3. Log1p transformation (ln(x + 1))
        result_df[f'{col}_log1p'] = np.log1p(abs_values)
        
        # 4. Signed log transformation (maintains sign of original data)
        result_df[f'{col}_signed_log'] = signs * np.log(abs_values + epsilon)
        
        # 5. Box-Cox-like transformation for positive values
        result_df[f'{col}_boxcox'] = np.zeros_like(col_values)
        positive_mask = col_values > 0
        result_df.loc[positive_mask, f'{col}_boxcox'] = np.log(result_df.loc[positive_mask, col])
        
        # 6. Symmetrical log transformation
        result_df[f'{col}_symlog'] = signs * np.log1p(abs_values)
        
        # Power transformations
        # 7. Square root transformation (preserving signs)
        result_df[f'{col}_sqrt'] = signs * np.sqrt(abs_values)
        
        # 8. Cube root transformation (handles negative values naturally)
        result_df[f'{col}_cbrt'] = np.cbrt(col_values)
        
        # 9. Square transformation
        result_df[f'{col}_square'] = np.square(col_values)
        
        # 10. Inverse transformation (1/x)
        with np.errstate(divide='ignore', invalid='ignore'):
            inverse_values = np.where(abs_values < epsilon, 
                                    signs * (1/epsilon), 
                                    signs * (1/(abs_values + epsilon)))
        result_df[f'{col}_inverse'] = inverse_values
        
        # 11. Yeo-Johnson transformation
        lambda_param = 0.5
        pos_mask = col_values >= 0
        neg_mask = ~pos_mask
        
        result_df[f'{col}_yeojohnson'] = np.zeros_like(col_values)
        
        # Transform positive values
        pos_values = ((np.power(col_values[pos_mask] + 1, lambda_param) - 1) / lambda_param)
        result_df.loc[pos_mask, f'{col}_yeojohnson'] = pos_values
        
        # Transform negative values
        neg_values = -(np.power(-col_values[neg_mask] + 1, 2-lambda_param) - 1) / (2-lambda_param)
        result_df.loc[neg_mask, f'{col}_yeojohnson'] = neg_values
        
    return result_df

