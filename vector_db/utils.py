import os
import json
import numpy as np

def ensure_folder_exists(folder_path: str):
    """
    Ensures that a folder exists, creating it if necessary.

    Args:
        folder_path: Path to the folder.
    """
    os.makedirs(folder_path, exist_ok=True)

def load_json(file_path: str):
    """
    Loads a JSON file and returns its content.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The parsed JSON content as a dictionary or list.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, "r") as file:
        return json.load(file)

def save_json(data, file_path: str):
    """
    Saves data to a JSON file.

    Args:
        data: Data to save (dict or list).
        file_path: Path to the JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def load_npy(file_path: str):
    """
    Loads a .npy file and returns its content.

    Args:
        file_path: Path to the .npy file.

    Returns:
        The content of the .npy file as a numpy array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    return np.load(file_path)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a numpy vector.

    Args:
        vector: Input numpy array.

    Returns:
        Normalized numpy array.
    """
    return vector / np.linalg.norm(vector)
