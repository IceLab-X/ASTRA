import torch
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import logging

# Parameter precision setting
torch.set_default_dtype(torch.double)


def calculate_mutual_information(X: np.ndarray, Y: np.ndarray, n_neighbors: int = 3, n_repeats: int = 10,
                                 logger=None) -> np.ndarray:
    """
    Calculates the mutual information (MI) between input parameters (X) and output metrics (Y).

    Args:
        X: Input data (log space, N x Input_Dim).
        Y: Output data (transformed for GP, N x Output_Dim).

    Returns:
        np.ndarray: Average MI results (Output_Dim x Input_Dim).
    """
    try:
        n_samples, n_inputs = X.shape
        n_outputs = Y.shape[1]
        mi_avg = np.zeros((n_outputs, n_inputs))  # Initialize average MI array

        for repeat in range(n_repeats):
            for output_index in range(n_outputs):
                # Mutual information regression calculation
                mi = mutual_info_regression(X, Y[:, output_index], n_neighbors=n_neighbors)
                mi_avg[output_index] += mi

        mi_avg /= n_repeats  # Calculate average mutual information values

        return mi_avg
    except Exception as e:
        if logger:
            logger.error(f"Error calculating Mutual Information: {e}")
        # Return a zero array to prevent crash
        return np.zeros((Y.shape[1], X.shape[1]))


def filter_two_rows(y_log_space: torch.Tensor, logger=None) -> tuple:
    """
    Filters and transforms the Y data (performance metrics) for Gaussian Process (GP) training.

    This function applies the constraints internally to calculate a FoM for the GP,
    which guides the optimization towards the constrained region.

    y_log_space is the raw output from the simulation function (log scale or initial transform)

    Returns: (modified_y_tensor (FoM for GP), FoM_count, gain_count, I_count, phase_count, GBW_count)
    """

    modified_y = []
    FoM_num = 1

    # Initialize metric counts (start at 1 to align with optimization stage counters)
    gain_num = 1
    i_num = 1
    phase_num = 1
    gbw_num = 1

    # Define hardcoded constraints for filtering (consistent with the original MI_calc.py)
    # Target Constraints (Real Space)
    target_gain = 60
    target_current = 1e-3  # 1mA
    current_multiplier = 1.8
    target_phase = 60
    target_gbw = 4e6

    # Create a detached copy for transformation
    y_alt = y_log_space.clone().detach()

    # Apply reverse log transform (partially to real space for constraint checking)
    y_alt[:, 0] = y_alt[:, 0] * 20  # Gain (linear -> dB)
    y_alt[:, 1] = -y_alt[:, 1]  # Current (negative log -> positive log)
    y_alt[:, 1:] = torch.exp(y_alt[:, 1:])  # Current, Phase, GBW (log -> exp)

    # Iterate over rows for FoM calculation and counting
    for i, row in enumerate(y_alt):
        # Check if constraints are satisfied
        is_feasible = (row[0] > target_gain and
                       row[1] * current_multiplier < target_current and
                       row[2] > target_phase and
                       row[3] > target_gbw)

        if is_feasible:
            # FoM value if constraints are met: 3 + f(Current) * weight
            # This incentivizes lower current within the feasible region
            modified_value = 3 + (target_current / row[1]) * 50
            FoM_num += 1
            modified_y.append(modified_value)
        else:
            # FoM value if constraints are NOT met: 0 (This is a simplified approach
            # used in the original code for early filtering/ranking)
            modified_value = 0
            modified_y.append(modified_value)

        # Count individual metric satisfaction
        gain_num += (row[0] >= target_gain)
        i_num += (row[1] * current_multiplier <= target_current)
        phase_num += (row[2] >= target_phase)
        gbw_num += (row[3] >= target_gbw)

    # Convert results to tensor
    modified_y = torch.tensor(modified_y, dtype=torch.double).unsqueeze(1)

    if logger:
        logger.info(f"filter_two_rows: FoM Count: {FoM_num}, Feasibility Y shape: {modified_y.shape}")

    # Return filtered Y for GP training and metric counts
    return modified_y, FoM_num, gain_num, i_num, phase_num, gbw_num


def calculate_scores(dbx: torch.Tensor, dby: torch.Tensor, FoM_num: int, I_num: int,
                     gain_num: int, GBW_num: int, phase_num: int, iter: int, init_num: int,
                     n_neighbors: int = 5, n_repeats: int = 100, input_dim: int = 12, logger=None) -> torch.Tensor:
    """
    Calculates the parameter sensitivity scores based on Mutual Information (MI),
    using dynamic weights based on constraint satisfaction rates.
    """

    # --- Dynamic Weight Calculation (based on ASTRA paper's implied logic) ---
    total_points = iter + init_num + 1  # Total number of evaluated points

    # Calculate weight based on the proportion of points *failing* the constraint
    gain_weight = (total_points - gain_num) / total_points if total_points > 0 else 0
    I_weight = (total_points - I_num) / total_points if total_points > 0 else 0
    GBW_weight = (total_points - GBW_num) / total_points if total_points > 0 else 0
    phase_weight = (total_points - phase_num) / total_points if total_points > 0 else 0
    # FoM weight is emphasized (proportion failing + 1)
    FoM_weight = ((total_points - FoM_num) / total_points if total_points > 0 else 0) + 1

    if logger:
        logger.info(
            f"MI Weights - Gain: {gain_weight:.2f}, Current: {I_weight:.2f}, Phase: {phase_weight:.2f}, GBW: {GBW_weight:.2f}, FoM: {FoM_weight:.2f}")

    # Convert tensors to numpy for sklearn MI calculation
    if isinstance(dbx, torch.Tensor):
        dbx_np = dbx.detach().numpy()
    if isinstance(dby, torch.Tensor):
        dby_np = dby.detach().numpy()

    # Calculate MI results (N_outputs x N_inputs)
    # dby for MI should have 5 columns: gain, current, phase, gbw, FoM
    if dby_np.shape[1] != 5:
        if logger:
            logger.error(
                f"Expected 5 columns in 'dby' for MI calculation, but got {dby_np.shape[1]}. Returning zero scores.")
        return torch.zeros(input_dim, dtype=torch.double)

    mi_results = calculate_mutual_information(dbx_np, dby_np, n_neighbors=n_neighbors, n_repeats=n_repeats,
                                              logger=logger)

    # Use the calculated dynamic weights
    weights = [gain_weight, I_weight, phase_weight, GBW_weight, FoM_weight]

    def cal_score(mi_results: np.ndarray, index: int, weights: list) -> float:
        """Calculates the weighted score for a single input parameter."""
        # Index corresponds to the input parameter index (0 to 11)
        # i corresponds to the output metric index (0 to 4: gain, current, phase, gbw, FoM)
        # Ensure dimensions match before summing
        score = 0
        num_outputs = mi_results.shape[0]
        if len(weights) == num_outputs:
            score = sum(mi_results[i][index] * weight for i, weight in enumerate(weights))
        else:
            if logger:
                logger.warning(
                    f"Mismatch between number of MI outputs ({num_outputs}) and weights ({len(weights)}). Cannot calculate score for index {index}.")
        return score

    indices = range(input_dim)
    scores_list = [cal_score(mi_results, i, weights) for i in indices]

    scores = torch.tensor(scores_list, dtype=torch.double)

    if logger:
        for i, mi in enumerate(mi_results):
            metric_name = ["Gain", "Current", "Phase", "GBW", "FoM"][i]
            logger.info(f"MI (Output {metric_name}): {mi}")

    return scores

