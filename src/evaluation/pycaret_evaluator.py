import gc  # Garbage collector to free memory after each fold
import json
import os
from time import strftime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

import ray  # Import Ray
from pycaret.classification import (create_model, predict_model, pull, save_model,
                                    setup, tune_model)

os.environ["RAY_DEDUP_LOGS"] = "0"


class PyCaretEvaluator:
    """
    Class to evaluate models using PyCaret, optimized for memory management.
    """

    def __init__(self, dataset: Any, target: str, experiment_name: Optional[str], filepath: str, columns: Optional[List[str]] = None):
        """
        Initialize the class parameters.

        Args:
            dataset (Any): the used dataset for machine learning.
            target (str): the target class.
            experiment_name (Optional[str]): optional name for the experiment.
            filepath (str): path for saving results.
            columns (Optional[List[str]]): optional list of column names.
        """
        self.dataset = dataset
        self.target = target
        self.experiment_name = experiment_name if experiment_name else f"experiment_{strftime('%Y%m%d-%H%M%S')}"
        self.filepath = filepath
        self.columns = columns

        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def save_results(self, results: List[Dict], filename: str) -> None:
        """
        Save the results in a JSON file.

        Args:
            results (List[Dict]): results to save.
            filename (str): file name where the results are saved.
        """
        with open(os.path.join(self.filepath, filename), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    @ray.remote(memory=6e9)  # Mark this function to be executed in parallel by Ray
    def run_fold(self, 
                 train_index: np.ndarray, 
                 test_index: np.ndarray, 
                 fold_num: int, 
                 train_size: float, 
                 fold: int, 
                 fold_strategy: str, 
                 session_id: int, 
                 model: str, 
                 optimize: Union[str, List[str]], 
                 custom_grid: Optional[Dict[str, List[Any]]], 
                 search_algorithm: str, 
                 fixed_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single fold in parallel using Ray.

        Args:
            train_index (np.ndarray): Array of training indices for the fold.
            test_index (np.ndarray): Array of testing indices for the fold.
            fold_num (int): The current fold number.
            train_size (float): Proportion of data to use for training within the fold.
            fold (int): Number of folds for inner cross-validation.
            fold_strategy (str): Strategy for inner cross-validation (e.g., 'kfold', 'stratifiedkfold').
            session_id (int): Random seed for reproducibility.
            model (str): Name of the model to be created (e.g., 'xgboost', 'lightgbm').
            optimize (Union[str, List[str]]): Metric(s) to optimize during model tuning (e.g., 'AUC', 'Accuracy').
            custom_grid (Optional[Dict[str, List[Any]]]): Custom hyperparameter grid for tuning the model.
            search_algorithm (str): Hyperparameter search algorithm to use ('grid' or 'random').
            fixed_params (Dict[str, Any]): Dictionary of fixed parameters for model setup (e.g., seed, eval_metric).

        Returns:
            Dict[str, Any]: A dictionary containing training results, test predictions, and the best hyperparameters for the fold.
        """
        print(f"Outer fold {fold_num}")

        # Extract the subsets using the given indices from StratifiedKFold
        train_data_x = self.dataset.x[train_index]
        train_data_y = self.dataset.y[train_index]
        test_data_x = self.dataset.x[test_index]
        test_data_y = self.dataset.y[test_index]

        # PyCaret setup expects a DataFrame, so we convert the NumPy arrays to DataFrames
        train_df = pd.DataFrame(train_data_x, columns=self.columns)
        train_df[self.target] = train_data_y
        test_df = pd.DataFrame(test_data_x, columns=self.columns)
        test_df[self.target] = test_data_y

        print(f"Train indices: {train_index}")
        print(f"Test indices: {test_index}")

        # Configure PyCaret for the current fold
        exp = setup(data=train_df,
                    target=self.target,
                    train_size=train_size,
                    fold=fold,
                    fold_strategy=fold_strategy,  # Directly using StratifiedKFold internally if needed
                    session_id=fixed_params['seed'],
                    verbose=False,
                    n_jobs=1)

        print(f"Configuring PyCaret for outer fold {fold_num}")

        # Create and tune the specified model
        best_model = create_model(model, fold=fold)

        if custom_grid:
            print(f"Tuning hyperparameters for model {model} with custom grid using {search_algorithm} search")
            best_model = tune_model(best_model, custom_grid=custom_grid, fold=fold, optimize=optimize, search_algorithm=search_algorithm, verbose=False)

            # Extract the best hyperparameters after tuning
            best_hyperparams = best_model.get_params()
        else:
            best_hyperparams = best_model.get_params()  # Default parameters if no tuning

        # Get the results and predictions
        model_results = pull()  # Pull the results after create_model or tune_model
        save_model(best_model, os.path.join(self.filepath, f"best_model_fold_{fold_num}"))
        test_predictions = predict_model(best_model, data=test_df)

        # Save the results of the fold
        split_result = {
            'fold': fold_num,
            'train_results': model_results.to_dict(),
            'test_predictions': test_predictions.to_dict(),
            'best_hyperparams': best_hyperparams  # Save best hyperparameters for this fold
        }

        # Clean up memory after each fold
        del train_df, test_df, best_model, model_results, test_predictions, exp
        gc.collect()

        return split_result

    def run_experiment(self, 
                       train_size: float = 0.8, 
                       fold: int = 5, 
                       fold_strategy: str = 'kfold', 
                       outer_fold: int = 5, 
                       outer_strategy: str = 'stratifiedkfold',
                       session_id: int = 42, 
                       model: Optional[str] = 'xgboost', 
                       optimize: Union[str, List[str]] = 'AUC', 
                       custom_grid: Optional[Dict[str, List[Any]]] = None,
                       search_algorithm: str = 'grid', 
                       fixed_params: Dict[str, Any] = None) -> None:
        """
        Executes the complete experiment including external cross-validation, training, and model optimization.

        Args:
            train_size (float): Proportion of the dataset to include in the training split.
            fold (int): Number of folds for internal cross-validation.
            fold_strategy (str): Strategy for internal cross-validation ('kfold', 'stratifiedkfold').
            outer_fold (int): Number of folds for external cross-validation.
            outer_strategy (str): Strategy for external cross-validation ('kfold', 'stratifiedkfold', 'random_sampling').
            session_id (int): Session ID for reproducibility.
            model (Optional[str]): Specific model to use.
            optimize (Union[str, List[str]]): The metric to optimize.
            custom_grid (Optional[Dict[str, List[Any]]]): Custom grid of parameters for tuning.
            search_algorithm (str): Algorithm to use for hyperparameter tuning ('grid' or 'random').
            fixed_params (Dict[str, Any]): Fixed parameters such as seed and eval_metric.
        """
        if fixed_params is None:
            fixed_params = {'seed': 42, 'eval_metric': 'logloss', 'verbosity': 0}

        # Define the outer cross-validation strategy
        if outer_strategy == 'stratifiedkfold':
            outer_cv = StratifiedKFold(n_splits=outer_fold, shuffle=True, random_state=session_id)
        elif outer_strategy == 'kfold':
            outer_cv = KFold(n_splits=outer_fold, shuffle=True, random_state=session_id)
        else:
            raise ValueError(f"Unknown outer_strategy: {outer_strategy}")
        
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
        ray_tasks = []  # List to store Ray tasks

        for i, (train_index, test_index) in enumerate(outer_cv.split(self.dataset.x, self.dataset.y)):
            ray_task = self.run_fold.remote(self, train_index, test_index, i + 1, train_size, fold, fold_strategy, session_id, model, 
                                            optimize, custom_grid, search_algorithm, fixed_params)
            ray_tasks.append(ray_task)

        results = ray.get(ray_tasks)
        self.save_results(results, f"{self.experiment_name}_results.json")

        # Collect and compute final metrics after training
        fold_metrics_list = []
        best_hyperparams_list = []

        for result in results:
            train_results = result.get('train_results', {})
            best_hyperparams = result.get('best_hyperparams', {})
            if isinstance(train_results, dict):
                metrics_df = pd.DataFrame(train_results, index=[0])
                fold_metrics_list.append(metrics_df)
            if best_hyperparams:
                best_hyperparams_list.append(best_hyperparams)

        if fold_metrics_list:
            all_fold_metrics = pd.concat(fold_metrics_list, ignore_index=True)
            final_metrics_mean = all_fold_metrics.mean()
            final_metrics_std = all_fold_metrics.std()

            metrics_table = pd.DataFrame({'Metric': final_metrics_mean.index, 'Mean': final_metrics_mean.values, 
                                          'Std Dev': final_metrics_std.values})
            print("Final metrics table:")
            print(metrics_table)

            metrics_table.to_csv(os.path.join(self.filepath, f"{self.experiment_name}_final_metrics.csv"), index=False)

        # Selecting the best hyperparameters based on the fold results
        if best_hyperparams_list:
            # You can either calculate the most frequent hyperparameters or take an average depending on the nature
            best_hyperparams_df = pd.DataFrame(best_hyperparams_list)
            most_common_hyperparams = best_hyperparams_df.mode().iloc[0]  # Most frequent hyperparameters across folds
            print(f"Best hyperparameters across all folds: {most_common_hyperparams}")

        ray.shutdown()
