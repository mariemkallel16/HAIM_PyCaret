"""
Filename: run_experiments.py

Author : Hakima Laribi, Mariem Kallel

Description: This file is used to perform all the HAIM experiments presented
             in the paper: https://doi.org/10.1038/s41746-022-00689-4

Date of last modification : 2023/02/07
"""

import argparse
from itertools import combinations
from tqdm import tqdm
from typing import List, Optional

from numpy import unique
from pandas import read_csv, DataFrame

from src.data import constants
from src.data.dataset import HAIMDataset
from src.evaluation.pycaret_evaluator import PyCaretEvaluator


def get_all_sources_combinations(sources: List[str]) -> List[List[str]]:
    """
    Function to extract all possible combinations of sources

    Args:
        sources(List[str]): list of sources types

    Returns: list of combinations
    """
    comb = []
    for i in range(len(sources)):
        combination = combinations(sources, i + 1)
        for c in combination:
            comb.append(list(c))

    return comb


def run_single_experiment(prediction_task: str,
                          sources_predictors: List[str],
                          sources_modalities: List[str],
                          dataset: Optional[DataFrame] = None,
                          evaluation_name: Optional[str] = None) -> None:
    """
    Function to perform one single experiment

    Args:
        prediction_task(task): task label, must be a HAIM prediction task
        sources_predictors(List[str]): predictors to use for prediction, each source has one or more predictors
        sources_modalities(List[str]): the modalities of the sources used for prediction
        dataset(Optional[DataFrame]): HAIM dataframe
        evaluation_name(Optional[str]): name of the experiment
    """
    dataset = read_csv(constants.FILE_DF, nrows=constants.N_DATA) if dataset is None else dataset

    # Create the HAIMDataset
    dataset = HAIMDataset(dataset,
                          sources_predictors,
                          sources_modalities,
                          prediction_task,
                          constants.IMG_ID,
                          constants.GLOBAL_ID)

    # Define the grid of hyper-parameters for the tuning
    grid_hps = {'max_depth': [5, 6, 7, 8],
                'n_estimators': [200, 300],
                'learning_rate': [0.3, 0.1, 0.05]}

    # Initialize the PyCaret Evaluator
    evaluator = PyCaretEvaluator(dataset=dataset,
                                 target=prediction_task,  # Ajout de l'argument target ici
                                 experiment_name=evaluation_name,
                                 filepath="constants.EXPERIMENT_PATH")

    # Model training and results evaluation
    evaluator.run_experiment(
        train_size=0.8,
        fold=5,
        fold_strategy='kfold',
        outer_fold=5,
        outer_strategy='kfold',
        session_id=42,
        model='xgboost',
        optimize='AUC',
        custom_grid=grid_hps
    )


if __name__ == '__main__':
    # Get arguments passed
    parser = argparse.ArgumentParser(description='Select a specific task')
    parser.add_argument('-t', '--task', help='prediction task to evaluate through all sources combinations',
                        default=None, dest='task')
    args = parser.parse_args()

    # Load the dataframe from disk
    df = read_csv(constants.FILE_DF, nrows=constants.N_DATA)

    # Handle all tasks if none specified
    all_tasks = [args.task] if args.task else Task()

    for task in all_tasks:
        print("#" * 23, f"{task} experiment", "#" * 23)

        # Get all possible combinations of sources for the current task
        sources_comb = get_all_sources_combinations(constants.SOURCES) if task in [constants.MORTALITY, constants.LOS] \
            else get_all_sources_combinations(constants.CHEST_SOURCES)

        with tqdm(total=len(sources_comb)) as bar:
            for count, combination in enumerate(sources_comb):
                # Get all predictors and modalities for each source
                predictors = []
                for c in combination:
                    predictors.extend(c.sources)  # Collect all predictors
                modalities = unique([c.modality for c in combination])

                # Run the single experiment
                run_single_experiment(prediction_task=task,
                                      sources_predictors=predictors,
                                      sources_modalities=modalities,
                                      dataset=df,
                                      evaluation_name=task + '_' + str(count))
                bar.update()

        # Optionally, collect the best experiments
        PyCaretEvaluator.get_best_of_experiments(task, constants.EXPERIMENT_PATH, count)
