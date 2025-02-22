import os
import numpy as np
import pandas as pd
import copy
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV


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


def perform_randomized_search_cv(
    base_pipeline,
    model_list_with_hyperparameters,
    X,
    y,
    scoring,
    cv,
    report_path,
    n_iter=10,
    random_state=42
):
    """
    Perform randomized hyperparameter search (with cross-validation) for each model in 
    `model_list_with_hyperparameters` using a common preprocessing pipeline. Each model 
    (plus its hyperparameters) is wrapped in the pipeline to avoid data leakage, and 
    every hyperparameter combination tested (up to `n_iter` draws) is recorded in an 
    Excel file with summary statistics.

    This function:
      1. Wraps preprocessing steps + model in a single Pipeline.
      2. Uses RandomizedSearchCV to search over the specified hyperparameter distributions.
      3. Extracts the mean, std, and median of each scoring metric for every hyperparameter 
         combination tried during the random search.
      4. Saves the results in an Excel spreadsheet.

    Args:
        base_pipeline (Pipeline):
            A scikit-learn Pipeline containing all preprocessing steps **except** 
            the final estimator. For example:
                Pipeline(steps=[
                    ('scaler', RobustScaler()),
                    ('under', RandomUnderSampler()),
                ])
            These steps are applied fold-by-fold within the cross-validation, 
            preventing data leakage.
        model_list_with_hyperparameters (list of dict):
            A list of dictionaries. Each dictionary has exactly one key-value pair:
            { model_instance: hyperparameter_distributions }, where
            `model_instance` is a scikit-learn estimator (e.g., LogisticRegression()) 
            and `hyperparameter_distributions` is a dict mapping parameter names to 
            lists/ranges of possible values (for RandomizedSearchCV).
            Example:
                [
                  {
                    RandomForestClassifier(): {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        ...
                    }
                  },
                  {
                    LogisticRegression(): {
                        'C': [0.001, 0.01, 0.1, 1],
                        'penalty': ['l1', 'l2'],
                        ...
                    }
                  }
                ]
        X (pd.DataFrame or np.ndarray):
            Feature matrix for training.
        y (pd.Series or np.ndarray):
            Target array for training.
        scoring (str, list, dict or callable):
            Metrics to evaluate on the test set. Can be:
              - A single string (e.g., 'accuracy'),
              - A list of strings (e.g., ['accuracy', 'f1']),
              - A dictionary mapping metric names to callables,
              - A single callable, or a list of callables.
            Refer to scikit-learn's `RandomizedSearchCV` and `make_scorer` documentation for details.
        cv (int or cross-validation generator):
            Determines the cross-validation splitting strategy. For example:
                RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1).
        report_path (str):
            Path to the output Excel file where results will be saved.
        n_iter (int, optional):
            Number of parameter settings that are sampled in RandomizedSearchCV. 
            Defaults to 10.
        random_state (int, optional):
            Controls the randomization of the algorithm. Defaults to 42.

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

        >>> # Define models with hyperparameter grids
        >>> model_list_with_hyperparameters = [
        ...     {
        ...         DecisionTreeClassifier(): {
        ...             'max_depth': [1, 2, 3, None],
        ...             'min_samples_leaf': [1, 2]
        ...         }
        ...     },
        ...     {
        ...         LogisticRegression(): {
        ...             'C': [0.01, 0.1, 1],
        ...             'penalty': ['l2']
        ...         }
        ...     }
        ... ]

        >>> # Define cross-validation
        >>> cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)

        >>> # Define scoring
        >>> scoring = ['accuracy', 'f1']

        >>> # Define output path
        >>> report_path = 'class_weighted_models.xlsx'

        >>> # Run cross-validation with randomized hyperparameter search
        >>> perform_randomized_search_cv(
        ...     base_pipeline=base_pipeline,
        ...     model_list_with_hyperparameters=model_list_with_hyperparameters,
        ...     X=X_train,
        ...     y=y_train,
        ...     scoring=scoring,
        ...     cv=cv,
        ...     report_path=report_path,
        ...     n_iter=5,
        ...     random_state=42
        ... )
        The file was successfully saved to class_weighted_models.xlsx
    """
    # Convert scoring to a list if it's a single string or a single callable
    if isinstance(scoring, str) or callable(scoring):
        scoring = [scoring]

    # If scoring is a dict, extract the metric names from the keys.
    # If it's a list, those are the metric names (strings or callables).
    if isinstance(scoring, dict):
        metric_names = list(scoring.keys())
    else:
        metric_names = []
        for s in scoring:
            if callable(s):
                metric_names.append(s.__name__ if hasattr(s, '__name__') else str(s))
            else:
                metric_names.append(s)

    all_results = []

    # Iterate through each model + hyperparameter grid combination
    for model_dict in model_list_with_hyperparameters:
        # Each dict is expected to have exactly one {model: param_dist} pair
        for model, param_dist in model_dict.items():
            # Create a fresh pipeline with the final step as the current model
            pipeline_steps = copy.deepcopy(base_pipeline.steps) + [('model', model)]
            pipeline_with_model = Pipeline(steps=pipeline_steps)

            # Adjust hyperparameter dict to reference the pipeline's final step (model)
            # e.g. 'model__n_estimators' instead of 'n_estimators'
            param_dist_for_pipeline = {
                f"model__{key}": val for key, val in param_dist.items()
            }

            # Set up the randomized search
            search = RandomizedSearchCV(
                estimator=pipeline_with_model,
                param_distributions=param_dist_for_pipeline,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                refit=False,  # We only want CV scores, not a final refit
                return_train_score=False
            )

            # Run the search
            search.fit(X, y)

            # Extract cross-validation results for each hyperparameter setting tested
            cv_results = search.cv_results_

            # For multi-metric scoring, cv_results_ contains:
            #   mean_test_<metric>, std_test_<metric>, rank_test_<metric>,
            #   split0_test_<metric>, split1_test_<metric>, ...
            # plus a 'params' key with the hyperparameters used for that row.

            n_splits = cv.get_n_splits(X, y)  # For median calculation

            for i in range(len(cv_results['params'])):
                # Create a row for each hyperparameter combination
                row_data = {}
                row_data['Model'] = str(model)
                row_data['Hyperparameters'] = cv_results['params'][i]
                row_data['Cross Validation'] = str(cv)
                row_data['Pipeline'] = str(pipeline_with_model)

                # For each metric, store mean, std, median
                for metric in metric_names:
                    mean_key = f"mean_test_{metric}"
                    std_key = f"std_test_{metric}"

                    # RandomizedSearchCV automatically adds these columns if `scoring` is multi-metric
                    # or a single string. If scoring is a callable, it might use that callable's __name__.
                    if mean_key in cv_results:
                        row_data[f"{metric} Mean"] = cv_results[mean_key][i]
                        row_data[f"{metric} Std"] = cv_results[std_key][i] if std_key in cv_results else None

                        # Compute median from split columns, e.g., split0_test_<metric>, split1_test_<metric>, ...
                        split_cols = [col for col in cv_results.keys()
                                      if col.startswith("split") and col.endswith(f"_test_{metric}")]
                        # Gather scores for each split
                        split_scores = [cv_results[col][i] for col in split_cols]
                        row_data[f"{metric} Median"] = np.median(split_scores) if split_scores else None
                    else:
                        # If for some reason the metric columns are missing (e.g. custom callable naming),
                        # store None or placeholders
                        row_data[f"{metric} Mean"] = None
                        row_data[f"{metric} Std"] = None
                        row_data[f"{metric} Median"] = None

                all_results.append(row_data)

    # Convert collected results to a DataFrame
    df_results = pd.DataFrame(all_results)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Save the DataFrame to Excel
    df_results.to_excel(report_path, index=False)

    print(f"The file was successfully saved to {report_path}")
