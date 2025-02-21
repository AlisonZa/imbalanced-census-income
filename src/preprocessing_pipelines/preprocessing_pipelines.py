from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd


import pandas as pd

def common_data_preprocessing(df):
    """
    Preprocess the input dataframe by performing several transformations:
    
    1. Drop the 'education' column (redundant since 'education.num' encodes the same information).
    2. Encode nominal categorical features using one-hot encoding while avoiding multicollinearity 
       (i.e., using dummy encoding with drop_first=True).
    3. Create a new feature 'profit' as the difference between 'capital.gain' and 'capital.loss'.
    4. Create a new binary feature 'investor' indicating if the person has any capital gain 
       (1 if capital.gain > 0, else 0).
    5. Create a new binary feature 'american' indicating if the person is a U.S. citizen 
       (1 if native.country equals 'United-States', else 0).
    6. Map the 'income' column to binary labels (0 for '<=50K' and 1 for '>50K').
    7. Map the 'sex' column to binary labels (0 for 'Female' and 1 for 'Male').
    8. One-hot encode the categorical-nominal features: workclass, marital.status, occupation, 
       relationship, and race (excluding native.country, already encoded as 'american').
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the dataset with relevant columns.
    
    Returns
    -------
    pandas.DataFrame
        The preprocessed dataframe with new features and encoded categorical variables.
    """
    # Drop redundant column
    df.drop('education', axis=1, inplace=True)
    
    # Encode binary categorical variables
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
    
    # Create new features
    df['profit'] = df['capital.gain'] - df['capital.loss']
    df['investor'] = (df['capital.gain'] > 0).astype(int)
    df['american'] = (df['native.country'] == 'United-States').astype(int)
    
    # Drop the original 'native.country' column, because it will make our dataset much more sparse
    df.drop('native.country', axis=1, inplace=True)
    
    # List of categorical-nominal features to encode
    categorical_nominals = ['workclass', 'marital.status', 'occupation', 'relationship', 'race']
    
    # One-hot encode categorical-nominal features while avoiding multicollinearity
    df = pd.get_dummies(df, columns=categorical_nominals, drop_first=True, dtype=int)
    
    return df


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

