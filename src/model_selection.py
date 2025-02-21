import os
import numpy as np
import pandas as pd
import copy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

def cross_validate_models(base_pipeline, model_list, X, y, scoring, cv, report_path):
    """
    Perform cross-validation on a list of models using a common preprocessing pipeline,
    then save the summary statistics (mean, median, standard deviation of scores) to an Excel file.

    This function helps avoid data leakage by including all data transformations 
    (e.g., scaling, undersampling) inside the pipeline. Each fold in the cross-validation 
    will properly fit the transformers only on the training split.

    Args:
        base_pipeline (Pipeline):
            A scikit-learn Pipeline containing all preprocessing steps **except** the final estimator.
            For example:
                Pipeline(steps=[
                    ('scaler', RobustScaler()),
                    ('under', RandomUnderSampler()),
                ])
        model_list (list):
            A list of scikit-learn estimator instances (models) to be evaluated.
            For example:
                [
                    DecisionTreeClassifier(class_weight="balanced"), 
                    LogisticRegression(class_weight="balanced")
                ]
        X (pd.DataFrame or np.ndarray):
            Feature matrix for training.
        y (pd.Series or np.ndarray):
            Target array for training.
        scoring (list, dict or callable):
            Metrics to evaluate on the test set; can be a single string, 
            a list of strings, a dict mapping names to scorers, or a callable. 
            Refer to scikit-learn's `cross_validate` documentation for details.
        cv (int or cross-validation generator):
            Determines the cross-validation splitting strategy. 
            For example, RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1).
        report_path (str):
            Path to the output Excel file where results will be saved.

    Returns:
        None

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import RobustScaler
        >>> from imblearn.under_sampling import RandomUnderSampler
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import RepeatedStratifiedKFold
        >>> import pandas as pd

        >>> # Example data
        >>> X_train = pd.DataFrame([[0,1],[1,1],[2,3],[3,4]])
        >>> y_train = pd.Series([0,1,0,1])

        >>> # Define base pipeline (preprocessing only)
        >>> steps = [
        ...     ('scaler', RobustScaler()),
        ...     ('under', RandomUnderSampler()),
        ... ]
        >>> base_pipeline = Pipeline(steps=steps)

        >>> # Define models
        >>> model_list = [
        ...     DecisionTreeClassifier(class_weight='balanced'),
        ...     LogisticRegression(class_weight='balanced')
        ... ]

        >>> # Define cross-validation
        >>> cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)

        >>> # Define scoring
        >>> scoring = ['accuracy', 'f1']

        >>> # Define output path
        >>> report_path = 'class_weighted_models.xlsx'

        >>> # Run cross-validation
        >>> cross_validate_models(base_pipeline, model_list, X_train, y_train, scoring, cv, report_path)
        The file was successfully saved to class_weighted_models.xlsx
    """
    # We will collect the results in a list of dictionaries, one row per model
    results = []

    # Convert scoring to a list of metrics if it's a single string or callable
    if isinstance(scoring, str) or callable(scoring):
        scoring = [scoring]

    # If scoring is a dict, we need the specific names
    if isinstance(scoring, dict):
        metrics = list(scoring.keys())
    else:
        # Otherwise assume scoring is a list of strings/callables
        # For callables, we won't have a nice name, so we'll convert them to strings
        metrics = []
        for s in scoring:
            if callable(s):
                metrics.append(s.__name__ if hasattr(s, '__name__') else str(s))
            else:
                metrics.append(s)

    # For each model in the list, create a new pipeline by adding the model as the final step
    for model in model_list:
        # Copy the base pipeline steps and add the final estimator
        steps_with_model = copy.deepcopy(base_pipeline.steps) + [('model', model)]
        pipeline_with_model = Pipeline(steps=steps_with_model)

        # Run cross-validation
        cv_results = cross_validate(
            estimator=pipeline_with_model,
            X=X,
            y=y,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False
        )

        # Prepare a dictionary to store results for this model
        row_data = {
            'Model': str(model),
            'Pipeline': str(pipeline_with_model)
        }

        # For each metric, compute mean, median, and std
        # cross_validate returns keys like 'test_accuracy', 'test_f1', etc.
        for m in metrics:
            test_score_key = f'test_{m}'
            # Some metrics might not exist if callables were used as dict keys, so we check
            if test_score_key in cv_results:
                scores = cv_results[test_score_key]
                row_data[f'{m} Mean'] = np.mean(scores)
                row_data[f'{m} Std'] = np.std(scores)
                row_data[f'{m} Median'] = np.median(scores)
            else:
                # If the key doesn't exist, store None or a placeholder
                row_data[f'{m} Mean'] = None
                row_data[f'{m} Std'] = None
                row_data[f'{m} Median'] = None

        results.append(row_data)

    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Save the DataFrame to Excel
    df_results.to_excel(report_path, index=False)

    print(f"The file was successfully saved to {report_path}")
