import random
import os
import numpy as np
import torch
import math
import pandas as pd

# Parameter precision setting
torch.set_default_dtype(torch.double)


def seed_set(seed: int, logger=None):
    """Fixes the random seed for reproducibility."""
    try:
        seed = int(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        if logger:
            logger.info(f"Random seed set to: {seed}")
        else:
            print(f"Random seed set to: {seed}")
    except Exception as e:
        if logger:
            logger.error(f"Error setting seed: {e}")
        else:
            print(f"Error setting seed: {e}")


def set_param_ranges(param_initial_value: float, percentage: float = 0.35, is_width_or_length: bool = False) -> list:
    """
    Calculates parameter min/max ranges based on an initial value and a percentage.
    Ensures minimum width/length constraints (0.18e-6) if applicable.
    """
    param_min = param_initial_value * (1 - percentage)
    param_max = param_initial_value * (1 + percentage)
    if is_width_or_length:
        # Ensure min is not less than technology minimum
        param_min = max(param_min, torch.tensor(0.18e-6, dtype=torch.double))
    return [param_min, param_max]


def set_log_bounds(value: float, percentage: float = 0.35) -> list:
    """Calculates log-space bounds based on an initial value and a percentage."""
    lower_bound = math.log(value * (1 - percentage))
    upper_bound = math.log(value * (1 + percentage))
    return [lower_bound, upper_bound]


def two_sort_and_group(var_name: list, scores: torch.Tensor) -> list:
    """Sorts parameters by score and groups them into sets of four for staged optimization."""
    # Convert scores tensor to a list of scores for zipping
    scores_list = scores.flatten().tolist()

    # Zip parameter names with their scores and sort descending
    param_scores = zip(var_name, scores_list)
    sorted_params = sorted(param_scores, key=lambda x: x[1], reverse=True)

    # Group into sets of four
    grouped_params = [(sorted_params[i], sorted_params[i + 1], sorted_params[i + 2], sorted_params[i + 3])
                      for i in range(0, len(sorted_params), 4)]
    return grouped_params


def get_indices_and_ranges(param_name: list, init_val: list, *params_percentage) -> tuple:
    """
    Retrieves the indices and real-space ranges for a subset of parameters.

    Args:
        param_name: List of all parameter names (e.g., ['cap', 'L1', ...]).
        init_val: List of current real-space values for all parameters.
        *params_percentage: Tuple of (parameter_name, percentage_range).

    Returns:
        Tuple: (list of indices, list of [min, max] ranges).
    """
    params_idx = []
    param_range = []
    for param, percentage in params_percentage:
        idx = param_name.index(param)
        params_idx.append(idx)
        # Check if parameter is a length/width (L or W)
        is_l_or_w = 'L' in param or 'W' in param
        param_range.append(set_param_ranges(init_val[idx], percentage, is_width_or_length=is_l_or_w))

    return params_idx, param_range


def ota_find_best(file_path: str, logger=None) -> list:
    """
    Finds the min and max of each performance metric from an initial CSV file.
    Used for calculating the Figure of Merit (FoM) in unconstrained optimization.
    """
    try:
        df = pd.read_csv(file_path)

        gain_max = df['gain(db)'].max()
        gain_min = df['gain(db)'].min()

        i_max = df['dc_current'].max()
        i_min = df['dc_current'].min()

        PM_max = df['phase'].max()
        PM_min = df['phase'].min()

        GBW_max = df['GBW(MHZ)'].max()
        GBW_min = df['GBW(MHZ)'].min()

        max_min = [gain_min, gain_max, i_min, i_max, PM_min, PM_max, GBW_min, GBW_max]
        return max_min
    except Exception as e:
        if logger:
            logger.error(f"Failed to read initial CSV file '{file_path}': {e}")
        raise e


def ota_two_fom_cal(y: torch.Tensor, min_max_list: list) -> float:
    """
    Calculates the Figure of Merit (FoM) for the OTA two-stage Op-Amp
    in unconstrained optimization mode.

    y = [Gain(dB), DC_Current(A), Phase Margin(deg), GBW(Hz)] (Real scale)
    min_max_list = [G_min, G_max, I_min, I_max, PM_min, PM_max, GBW_min, GBW_max]
    """
    # Define targets for clipping
    target_gain = 60
    target_current_limit = 1.0e-3 / 1.8  # Based on typical constraint
    target_phase = 60
    target_gbw = 4e6

    # Performance metrics (clamped to prevent optimization artifacts)
    g = min(y[0][0].item(), target_gain)
    i = min(y[0][1].item(), target_current_limit)
    pm = min(y[0][2].item(), target_phase)
    gbw = min(y[0][3].item(), target_gbw)

    # Min/Max values from the initial dataset
    G_min, G_max = min_max_list[0], min_max_list[1]
    I_min, I_max = min_max_list[2], min_max_list[3]
    PM_min, PM_max = min_max_list[4], min_max_list[5]
    GBW_min, GBW_max = min_max_list[6], min_max_list[7]

    # Calculate normalized FoM components
    # Maximized (Gain, PM, GBW): (Val - Min) / (Max - Min)
    # Minimized (Current): - (Val - Min) / (Max - Min)

    norm_gain = (g - G_min) / (G_max - G_min)
    norm_current = (i - I_min) / (I_max - I_min)
    norm_phase = (pm - PM_min) / (PM_max - PM_min)
    norm_gbw = (gbw - GBW_min) / (GBW_max - GBW_min)

    fom_value = norm_gain - norm_current + norm_phase + norm_gbw

    return fom_value
