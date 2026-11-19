from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "experiment_results"


def save_experiment_results(
        df: pd.DataFrame,
        dataset_name: str,
        measurement_type: str,
        use_timestamp: bool = False
) -> None:
    """
    Save experiment results DataFrame to a CSV file organized by measurement type.

    Args:
        df: DataFrame containing experiment results.
        dataset_name: Name of the dataset for file naming.
        measurement_type: Type of measurement; must be one of "diversity", "computation_time", or "representation_time".
        use_timestamp: If True, append a timestamp to the filename.

    Raises:
        ValueError: If measurement_type is invalid.
    """

    measurement_folders = {
        "diversity": "diversity",
        "computation_time": "computation_time",
        "representation_time": "representation_time"
    }

    if measurement_type not in measurement_folders:
        raise ValueError(f"Invalid measurement_type. Must be one of {list(measurement_folders.keys())}")

    folder_path = RESULTS_PATH / measurement_folders[measurement_type]
    folder_path.mkdir(parents=True, exist_ok=True)

    if use_timestamp:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{measurement_type}_{timestamp}.csv"
    else:
        filename = f"{dataset_name}_{measurement_type}.csv"

    file_path = folder_path / filename

    df.to_csv(file_path, index=True)

    print(f"Saved {measurement_type} results to {file_path}")


def load_experiment_results(
        dataset_name: str,
        measurement_type: str,
        identifier: str | int = 1
) -> pd.DataFrame:
    """
    Load experiment results from CSV file for a given dataset and measurement type.

    Args:
        dataset_name: Name of the dataset.
        measurement_type: Type of measurement; must be one of "diversity", "computation_time", or "representation_time".
        identifier: Either an integer index to select the latest file(s) or a string timestamp to specify the exact file.

    Raises:
        ValueError: If measurement_type is invalid.
        FileNotFoundError: If the file does not exist.
        IndexError: If the integer identifier is out of range.

    Returns:
        DataFrame containing the loaded experiment results.
    """

    measurement_folders = {
        "diversity": "diversity",
        "computation_time": "computation_time",
        "representation_time": "representation_time"
    }

    if measurement_type not in measurement_folders:
        raise ValueError(f"Invalid measurement_type. Must be one of {list(measurement_folders.keys())}")

    folder_path = RESULTS_PATH / measurement_folders[measurement_type]

    if isinstance(identifier, int):
        pattern = f"{dataset_name}_{measurement_type}_*.csv"
        matching_files = sorted(folder_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if not matching_files:
            raise FileNotFoundError(
                f"No results found for dataset '{dataset_name}' and measurement type '{measurement_type}'")
        if identifier < 1 or identifier > len(matching_files):
            raise IndexError(f"Invalid identifier index {identifier}. Only {len(matching_files)} file(s) found.")
        file_path = matching_files[identifier - 1]
    else:
        if identifier == "":
            expected_name = f"{dataset_name}_{measurement_type}.csv"
        else:
            expected_name = f"{dataset_name}_{identifier}_{measurement_type}.csv"
        file_path = folder_path / expected_name
        if not file_path.exists():
            raise FileNotFoundError(f"No file found with timestamp '{identifier}' for dataset '{dataset_name}'")

    print(f"Loading {measurement_type} results from {file_path}")
    return pd.read_csv(file_path, index_col=0)
