import os
import json
from time import strftime
from typing import Dict, List, Optional, Union, Any
from sklearn.model_selection import StratifiedKFold, KFold
from pycaret.classification import (
    create_model, setup, pull, save_model, predict_model, tune_model
)

import numpy as np
import pandas as pd
from src.data.dataset import HAIMDataset
import gc  # Garbage collector to liberate memory after each fold
 
 
class PyCaretEvaluator:

    """

    Class to evaluate models using PyCaret, optimized for memory management.

    """
 
    def __init__(self, dataset: HAIMDataset, target: str, experiment_name: Optional[str], filepath: str, columns: Optional[List[str]] = None):

        """
        Initialize the class parameters.
 
        Args:
            dataset (HAIMDataset): the used dataset for machine learning.
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

        # Define the fixed params
        self.fixed_params = {
            'seed': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
 
    def save_results(self, results: List[Dict], filename: str) -> None:

        """
        Save the results in a JSON file.
 
        Args:
            results (List[Dict]): results to save.
            filename (str): file name where the results are saved.
        """

        with open(os.path.join(self.filepath, filename), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
 
    def run_experiment(self,
                       train_size: float = 0.8,
                       fold: int = 5,
                       fold_strategy: str = 'kfold',
                       outer_fold: int = 5,
                       outer_strategy: str = 'stratifiedkfold',
                       session_id: int = 42,
                       model: Optional[str] = 'xgboost',
                       optimize: Union[str, List[str]] = 'AUC',
                       custom_grid: Optional[Dict[str, List[Any]]] = None) -> None:

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
        """

        # Define the outer cross-validation strategy
        if outer_strategy == 'stratifiedkfold':
            outer_cv = StratifiedKFold(n_splits=outer_fold, shuffle=True, random_state=session_id)
        elif outer_strategy == 'kfold':
            outer_cv = KFold(n_splits=outer_fold, shuffle=True, random_state=session_id)
        else:
            raise ValueError(f"Unknown outer_strategy: {outer_strategy}")
 
        results = []
 
        # External cross-validation loop

        for i, (train_index, test_index) in enumerate(outer_cv.split(self.dataset.x, self.dataset.y)):
            print(f"Outer fold {i + 1}/{outer_fold}")

            # Extract the subsets
            train_data_x = self.dataset.x[train_index]
            train_data_y = self.dataset.y[train_index]
            test_data_x = self.dataset.x[test_index]
            test_data_y = self.dataset.y[test_index]
 
            # Create DataFrames using specified columns
            train_df = pd.DataFrame(train_data_x, columns=self.columns)
            train_df[self.target] = train_data_y
            test_df = pd.DataFrame(test_data_x, columns=self.columns)
            test_df[self.target] = test_data_y
 
            # Configure PyCaret for the current fold with fixed parameters
            exp = setup(data=train_df,
                        target=self.target,
                        train_size=train_size,
                        fold=fold,
                        fold_strategy=fold_strategy,
                        session_id=self.fixed_params['seed'],  # Using the seed for reproducibility
                        verbose=False)  # Use verbose=False to reduce output during setup
 
            print(f"Configuring PyCaret for outer fold {i + 1}")
 
            # Create and tune the specified model
            best_model = create_model(model, fold=fold)

            if custom_grid:
                print(f"Tuning hyperparameters for model {model} with custom grid")
                best_model = tune_model(best_model, custom_grid=custom_grid, fold=fold)
 
            # Get the results and predictions
            model_results = pull()  # Pulls the results after create_model
            save_model(best_model, os.path.join(self.filepath, f"best_model_fold_{i}"))
            test_predictions = predict_model(best_model, data=test_df)
 
            # Save the results of the fold
            split_result = {
                'fold': i,
                'train_results': model_results.to_dict(),
                'test_predictions': test_predictions.to_dict()
            }

            results.append(split_result)
 
            # Clean up memory after each fold
            del train_df, test_df, best_model, model_results, test_predictions
            gc.collect()
 
        # Save results in a JSON file
        self.save_results(results, f"{self.experiment_name}_results.json")
 
        # Collect and compute final metrics after training
        fold_metrics_list = []
 
        for result in results:
            train_results = result.get('train_results', {})

            if isinstance(train_results, dict):

                # Convert the results for each fold into a DataFrame
                metrics_df = pd.DataFrame(train_results, index=[0])
                fold_metrics_list.append(metrics_df)
 
        if fold_metrics_list:

            # Combine the DataFrames from all folds
            all_fold_metrics = pd.concat(fold_metrics_list, ignore_index=True)

            # Calculate the mean and standard deviation for each metric across all folds
            final_metrics_mean = all_fold_metrics.mean()
            final_metrics_std = all_fold_metrics.std()
 
            # Create a summary table of the final metrics
            metrics_table = pd.DataFrame({
                'Metric': final_metrics_mean.index,
                'Mean': final_metrics_mean.values,
                'Std Dev': final_metrics_std.values
            })
 
            print("Final metrics table:")
            print(metrics_table)
 
            # Save the final metrics table to a CSV file
            metrics_table.to_csv(os.path.join(self.filepath, f"{self.experiment_name}_final_metrics.csv"), index=False)

 