"""
Filename: dataset_adapted.py
Author:   Adapted by Mariem Kallel from original code by Hakima Laribi (MEDomicsLab)
Description:
    Adapted version of HAIMDataset enabling flexible selection of
    modality subsets to reduce data volume while replicating the
    experiments from the HAIM paper (Soenksen et al., 2022).

Reference:
    Soenksen, L.R., Ma, Y., Zeng, C. et al. Integrated multimodal
    artificial intelligence framework for healthcare applications.
    npj Digit. Med. 5, 149 (2022).
    https://doi.org/10.1038/s41746-022-00689-4
"""

from typing import List, Union, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# ============================================================
# TASK CONSTANTS
# ============================================================
FRACTURE                   = 'Fracture'
LUNG_LESION                = 'Lung Lesion'
ENLARGED_CARDIOMEDIASTINUM = 'Enlarged Cardiomediastinum'
CONSOLIDATION              = 'Consolidation'
PNEUMONIA                  = 'Pneumonia'
LUNG_OPACITY               = 'Lung Opacity'
ATELECTASIS                = 'Atelectasis'
PNEUMOTHORAX               = 'Pneumothorax'
EDEMA                      = 'Edema'
CARDIOMEGALY               = 'Cardiomegaly'
MORTALITY                  = '48h mortality'
LOS                        = '48h los'

ALL_TASKS = [
    FRACTURE, LUNG_LESION, ENLARGED_CARDIOMEDIASTINUM, CONSOLIDATION,
    PNEUMONIA, LUNG_OPACITY, ATELECTASIS, PNEUMOTHORAX,
    EDEMA, CARDIOMEGALY, MORTALITY, LOS
]


# ============================================================
# SOURCE REGISTRY
# Maps source keys to embedding dimensions and column names.
# ============================================================
SOURCE_REGISTRY = {
    # ---- Tabular ----
    'de': {
        'n_embeddings': 6,
        'columns': [f"de_{i}" for i in range(6)],
        'modality': 'tabular',
        'description': 'Demographics',
    },
    # ---- Time-series ----
    'ts_ce': {
        'n_embeddings': 99,
        'columns': [f"ts_ce_{i}" for i in range(99)],
        'modality': 'time_series',
        'description': 'Chart events',
    },
    'ts_le': {
        'n_embeddings': 242,
        'columns': [f"ts_le_{i}" for i in range(242)],
        'modality': 'time_series',
        'description': 'Lab events',
    },
    'ts_pe': {
        'n_embeddings': 110,
        'columns': [f"ts_pe_{i}" for i in range(110)],
        'modality': 'time_series',
        'description': 'Procedure events',
    },
    # ---- Text ----
    'n_rad': {
        'n_embeddings': 768,
        'columns': [f"n_rad_{i}" for i in range(768)],
        'modality': 'text',
        'description': 'Radiology notes',
    },
    'n_ecg': {
        'n_embeddings': 768,
        'columns': [f"n_ecg_{i}" for i in range(768)],
        'modality': 'text',
        'description': 'ECG notes',
    },
    'n_ech': {
        'n_embeddings': 768,
        'columns': [f"n_ech_{i}" for i in range(768)],
        'modality': 'text',
        'description': 'Echocardiogram notes',
    },
    # ---- Images ----
    'vp': {
        'n_embeddings': 18,
        'columns': [f"vp_{i}" for i in range(18)],
        'modality': 'images',
        'description': 'Visual probabilities',
    },
    'vmp': {
        'n_embeddings': 18,
        'columns': [f"vmp_{i}" for i in range(18)],
        'modality': 'images',
        'description': 'Aggregated visual probabilities',
    },
    'vd': {
        'n_embeddings': 1024,
        'columns': [f"vd_{i}" for i in range(1024)],
        'modality': 'images',
        'description': 'Visual dense features',
    },
    'vmd': {
        'n_embeddings': 1024,
        'columns': [f"vmd_{i}" for i in range(1024)],
        'modality': 'images',
        'description': 'Aggregated visual dense features',
    },
}


# ============================================================
# PREDEFINED COMBINATIONS
# Inspired by the supplementary material of the HAIM paper.
# Format: preset_name -> list of SOURCE_REGISTRY keys
# ============================================================
PRESET_COMBINATIONS = {
    'tabular_only':          ['de'],
    'timeseries_only':       ['ts_ce', 'ts_le', 'ts_pe'],
    'tab_ts':                ['de', 'ts_ce', 'ts_le', 'ts_pe'],
    'tab_ts_ecg':            ['de', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg'],
    'tab_ts_ech':            ['de', 'ts_ce', 'ts_le', 'ts_pe', 'n_ech'],
    'tab_ts_ecg_ech':        ['de', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech'],
    'tab_ts_all_text':       ['de', 'ts_ce', 'ts_le', 'ts_pe', 'n_rad', 'n_ecg', 'n_ech'],
    'tab_ts_img_light':      ['de', 'ts_ce', 'ts_le', 'ts_pe', 'vp', 'vmp'],
    'tab_ts_img_light_text': ['de', 'ts_ce', 'ts_le', 'ts_pe', 'vp', 'vmp', 'n_ecg', 'n_ech'],
    'full_no_rad':           ['de', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech'],
    'full_haim':             ['de', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_rad', 'n_ecg', 'n_ech'],
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_sources_for_preset(preset_name: str) -> List[str]:
    """Return the list of source names for a given preset."""
    if preset_name not in PRESET_COMBINATIONS:
        raise ValueError(
            f"Unknown preset '{preset_name}'.\n"
            f"Available: {list(PRESET_COMBINATIONS.keys())}"
        )
    return PRESET_COMBINATIONS[preset_name]


def get_columns_for_sources(source_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Return (csv_columns, unique_modalities) for a list of source keys.

    Parameters
    ----------
    source_names : list of str
        E.g. ['de', 'ts_ce', 'vp']

    Returns
    -------
    columns : list of str
        Column names expected in the CSV.
    modalities : list of str
        Unique modalities present (first-occurrence order).
    """
    columns = []
    modalities = []
    seen_modalities: set = set()

    for src_name in source_names:
        if src_name not in SOURCE_REGISTRY:
            raise ValueError(
                f"Unknown source '{src_name}'.\n"
                f"Available: {list(SOURCE_REGISTRY.keys())}"
            )
        src = SOURCE_REGISTRY[src_name]
        columns.extend(src['columns'])
        if src['modality'] not in seen_modalities:
            modalities.append(src['modality'])
            seen_modalities.add(src['modality'])

    return columns, modalities


def describe_combination(source_names: List[str]) -> str:
    """Return a human-readable summary of a source combination."""
    total = 0
    lines = []
    for src_name in source_names:
        src = SOURCE_REGISTRY[src_name]
        total += src['n_embeddings']
        lines.append(
            f"  [{src['modality']:12s}] {src_name:8s} : "
            f"{src['n_embeddings']:5d} features  ({src['description']})"
        )
    lines.append(f"  {'':12s} {'TOTAL':8s} : {total:5d} features")
    return "\n".join(lines)


# ============================================================
# MAIN CLASS
# ============================================================

class Task:
    """Iterator over supported task names."""
    def __iter__(self):
        return iter(ALL_TASKS)


class HAIMDataset:
    """
    Adapted dataset class for HAIM experiments with flexible
    modality/source selection.

    Parameters
    ----------
    original_dataset : pd.DataFrame
        DataFrame loaded from the HAIM-MIMIC-MM CSV.
    task : str
        Prediction task (use the constants defined above).
    ids : str
        Column identifying unique observations (e.g. 'img_id').
    source_names : list of str, optional
        Source keys from SOURCE_REGISTRY to use.
        Required if `preset` is not provided.
    global_ids : str, optional
        Column for global patient identifier (e.g. 'haim_id').
    preset : str, optional
        Name of a predefined combination from PRESET_COMBINATIONS.
        Overrides `source_names` if provided.
    """

    def __init__(
        self,
        original_dataset: pd.DataFrame,
        task: str,
        ids: str,
        source_names: Optional[List[str]] = None,
        global_ids: Optional[str] = None,
        preset: Optional[str] = None,
    ):
        if preset is not None:
            source_names = get_sources_for_preset(preset)
        if source_names is None:
            raise ValueError("Provide either source_names or preset.")

        columns, modalities = get_columns_for_sources(source_names)

        for col in ['img_length_of_stay', 'death_status']:
            if col not in original_dataset.columns:
                raise ValueError(f"Required column missing: '{col}'")

        if task not in Task():
            raise ValueError(f"Unsupported task: '{task}'. Options: {ALL_TASKS}")

        missing = [c for c in columns if c not in original_dataset.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} columns missing from dataset "
                f"(e.g. {missing[:5]}). "
                f"Check that the requested sources exist in your CSV."
            )

        if ids not in original_dataset.columns:
            raise ValueError(f"ids column '{ids}' not found in dataset.")

        if global_ids is not None and global_ids not in original_dataset.columns:
            raise ValueError(f"global_ids column '{global_ids}' not found in dataset.")

        self._source_names = source_names
        self._sources      = columns
        self._modalities   = modalities
        self._original_dataset = original_dataset
        self._task         = task
        self._ids          = ids
        self._global_ids   = global_ids
        self._x = self._y = self._task_dataset = None

        print(f"\n{'='*55}")
        print(f" HAIMDataset  |  Task: {task}")
        print(f"{'='*55}")
        print(describe_combination(source_names))
        print()

        self.__create_dataset()

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def source_names(self) -> List[str]:
        return self._source_names

    @property
    def modalities(self) -> List[str]:
        return self._modalities

    @property
    def sources(self) -> List[str]:
        return self._sources

    @property
    def task(self) -> str:
        return self._task

    @property
    def ids(self) -> str:
        return self._ids

    @property
    def global_ids(self) -> Optional[str]:
        return self._global_ids

    @property
    def task_dataset(self) -> pd.DataFrame:
        return self._task_dataset

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def n_features(self) -> int:
        return self._x.shape[1] if self._x is not None else 0

    # ----------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------

    def __len__(self):
        return self._task_dataset.shape[0]

    def __getitem__(
        self,
        idx: Union[int, List[int], str, pd.DataFrame, pd.Series]
    ) -> Union[Tuple[np.ndarray, np.ndarray], pd.DataFrame, pd.Series]:
        if isinstance(idx, int):
            return self.x[idx], self.y[idx]
        if isinstance(idx, list) and isinstance(idx[0], int):
            return self.x[idx], self.y[idx]
        return self.task_dataset[idx]

    # ----------------------------------------------------------
    # Dataset construction
    # ----------------------------------------------------------

    def __create_dataset(self):
        df = self._original_dataset.copy()

        if self._task == MORTALITY:
            df.loc[
                (df['img_length_of_stay'] < 48) & (df['death_status'] == 1),
                self._task
            ] = 1
            df.loc[df['death_status'] == 0, self._task] = 0
            df.loc[
                (df['img_length_of_stay'] >= 48) & (df['death_status'] == 1),
                self._task
            ] = 0

        elif self._task == LOS:
            df.loc[
                (df['img_length_of_stay'] < 48) & (df['death_status'] == 0),
                self._task
            ] = 1
            df.loc[
                (df['img_length_of_stay'] < 48) & (df['death_status'] == 1),
                self._task
            ] = 0
            df.loc[df['img_length_of_stay'] >= 48, self._task] = 0

        mask = (df[self._task] == 0) | (df[self._task] == 1)
        self._task_dataset = df[mask].reset_index(drop=True)
        self._original_dataset = None  # free memory

        self._x = np.array(self._task_dataset[self._sources], dtype=float)
        self._y = np.array(self._task_dataset[self._task], dtype=int)

        n_pos   = int(self._y.sum())
        n_total = len(self._y)
        print(
            f" Samples: {n_total} | Positive: {n_pos} "
            f"({100*n_pos/n_total:.1f}%) | Features: {self.n_features}"
        )
        print(f"{'='*55}\n")

    # ----------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------

    def map_idx_to_global_ids(self) -> Optional[Dict[int, List[int]]]:
        if self._global_ids is None:
            return None
        map_ids = {}
        global_ids = self.task_dataset[self._global_ids].tolist()
        for id_ in global_ids:
            map_ids[id_] = self.task_dataset.index[
                self.task_dataset[self._global_ids] == id_
            ].tolist()
        return map_ids

    def map_idx_to_ids(self) -> Dict[int, List[int]]:
        map_ids = {}
        ids = self.task_dataset[self._ids].tolist()
        for id_ in ids:
            map_ids[id_] = self.task_dataset.index[
                self.task_dataset[self._ids] == id_
            ].tolist()
        return map_ids

    def get_global_ids(self, indexes: List[int]) -> Optional[List[int]]:
        if self.global_ids is None:
            return None
        return self.task_dataset.iloc[indexes][self.global_ids].unique().tolist()

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """Train/test split with optional stratification."""
        from sklearn.model_selection import train_test_split
        strat = self._y if stratify else None
        return train_test_split(
            self._x, self._y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    # ----------------------------------------------------------
    # MEDomics-optimized export
    # ----------------------------------------------------------

    def save_to_csv(
        self,
        filepath: str,
        drop_ids: bool = True,
        compress: bool = True,
        scale_factor: int = 100000,
        return_df: bool = False
    ):
        """
        Save a MEDomics-compatible CSV with optional memory compression.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        drop_ids : bool
            Drop identifier columns not needed by MEDomics.
        compress : bool
            Compress float features by scaling and casting to float32.
            float32 is used (rather than float16) to preserve NaN support.
        scale_factor : int
            Multiplication factor applied before rounding (default 100000).
        return_df : bool
            If True, return the exported DataFrame.
        """
        df = self._task_dataset.copy()

        required_cols = [self._task, 'death_status', 'img_length_of_stay']
        id_cols = ['haim_id', 'img_id', 'img_charttime']

        cols_to_keep = []
        for col in (self._sources + required_cols + id_cols):
            if col in df.columns and col not in cols_to_keep:
                cols_to_keep.append(col)

        df = df[cols_to_keep].copy()

        if compress:
            feature_cols = [c for c in self._sources if c in df.columns]
            float_cols = df[feature_cols].select_dtypes(include=['float']).columns.tolist()
            df[float_cols] = (df[float_cols] * scale_factor).round().astype("float32")

        if drop_ids:
            cols_to_drop = [c for c in ['img_id', 'img_charttime'] if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        df.to_csv(filepath, index=False)

        print("\nDataset saved:", filepath)
        print("Shape:", df.shape)
        print("Selected features:", len(self._sources))
        print("Estimated memory:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB\n")

        if return_df:
            return df


# ============================================================
# BATCH EXPERIMENT UTILITIES
# ============================================================

def run_experiment(
    df: pd.DataFrame,
    task: str,
    source_names: List[str],
    ids: str = 'img_id',
    global_ids: str = 'haim_id',
    test_size: float = 0.2,
    random_state: int = 42,
    n_runs: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Replicate a HAIM paper experiment: train XGBoost and return
    mean AUC over n_runs random splits.

    Parameters
    ----------
    df            : Full HAIM DataFrame.
    task          : Prediction task constant.
    source_names  : List of source keys to use.
    ids           : Unique observation identifier column.
    global_ids    : Global patient identifier column.
    test_size     : Test set proportion (default 0.2).
    random_state  : Base random seed, incremented per run.
    n_runs        : Number of repetitions (paper uses 5).
    verbose       : Print per-run AUC.

    Returns
    -------
    dict with keys: task, sources, aucs, mean_auc, std_auc
    """
    try:
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "Install missing dependencies:\n"
            "  pip install xgboost scikit-learn\n" + str(e)
        )

    dataset = HAIMDataset(
        original_dataset=df.copy(),
        task=task,
        ids=ids,
        source_names=source_names,
        global_ids=global_ids,
    )

    aucs = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = dataset.get_train_test_split(
            test_size=test_size,
            random_state=random_state + run,
        )
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state + run,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        aucs.append(auc)
        if verbose:
            print(f"  Run {run+1}/{n_runs}  AUC = {auc:.4f}")

    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    print(f"\n  → Mean: {mean_auc:.4f} ± {std_auc:.4f}\n")

    return {
        'task': task,
        'sources': source_names,
        'aucs': aucs,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
    }


def run_multiple_experiments(
    df: pd.DataFrame,
    tasks: List[str],
    combinations: List[List[str]],
    **kwargs,
) -> List[Dict]:
    """
    Run multiple experiments (tasks × combinations) and return all results.
    """
    results = []
    total = len(tasks) * len(combinations)
    for i, (task, combo) in enumerate(
        [(t, c) for t in tasks for c in combinations], 1
    ):
        print(f"\n[{i}/{total}] Task={task}  Sources={combo}")
        try:
            r = run_experiment(df=df, task=task, source_names=combo, **kwargs)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'task': task, 'sources': combo, 'error': str(e)})
    return results


# ============================================================
# USAGE EXAMPLES (uncomment to run)
# ============================================================

if __name__ == '__main__':

    # Print all presets and their feature counts
    print("=== Available presets ===\n")
    for name, srcs in PRESET_COMBINATIONS.items():
        print(f"[{name}]")
        print(describe_combination(srcs))
        print()

    # --- Load CSV ---
    # df = pd.read_csv('csvs/cxr_ic_fusion_1103.csv', low_memory=False)

    # --- Single experiment ---
    # results = run_experiment(
    #     df=df,
    #     task=ENLARGED_CARDIOMEDIASTINUM,
    #     source_names=['de', 'vp', 'vmp', 'ts_ce', 'ts_pe'],
    #     n_runs=5,
    # )

    # --- Multiple tasks x combinations ---
    # all_results = run_multiple_experiments(
    #     df=df,
    #     tasks=[ENLARGED_CARDIOMEDIASTINUM, CONSOLIDATION],
    #     combinations=[
    #         ['de', 'vp', 'vmp', 'ts_ce', 'ts_pe'],
    #         get_sources_for_preset('tab_ts_img_light'),
    #     ],
    #     n_runs=5,
    # )
