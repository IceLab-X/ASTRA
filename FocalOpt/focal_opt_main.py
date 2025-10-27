import random
import pandas as pd
import torch
import numpy as np
import time
import os
import csv
import logging
import traceback
import json
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Imports from modularized files
from .optimization_core import BayesianOptimization
from .utility_functions import seed_set, get_indices_and_ranges, two_sort_and_group
from .ota_config import init_OTA_two
from .mi_analysis import filter_two_rows, calculate_scores

# Parameter precision setting
torch.set_default_dtype(torch.double)

# --- LLM API Configuration ---
# Load environment variables from the .env file in the parent directory
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_MODEL = os.getenv("MODEL", "gpt-4o")  # Default to gpt-4o if not set

# Configure API details based on .env
if BASE_URL and not BASE_URL.endswith('/'):
    BASE_URL += '/'

# NOTE: The provided .env suggests an OpenAI-compatible API endpoint, you may modify base on it.
# We will use the V1 chat completion endpoint.
LLM_API_URL = f"{BASE_URL}v1/chat/completions"

MAX_RETRIES = 5


# --- End LLM API Configuration ---

def format_data_for_prompt(dbx: torch.Tensor, dby_real: List[List[float]], param_names: List[str],
                           num_samples: int = 5) -> str:
    """Formats a small sample of the design data (X and Y) for the LLM prompt."""

    # dbx is log space, convert back to real scale for LLM
    dbx_real = torch.exp(dbx).cpu().numpy()

    # Combine X (real) and Y (real) into readable strings
    data_list = []

    # Ensure num_samples doesn't exceed the available data size
    num_samples = min(num_samples, len(dbx_real))

    # Select a small, representative sample of points
    indices = np.linspace(0, len(dbx_real) - 1, num_samples, dtype=int)

    for i in indices:
        x_str = ", ".join([f"{name}: {val:.2e}" for name, val in zip(param_names, dbx_real[i])])
        y_str = f"Gain: {dby_real[i][0]:.2f}dB, Current: {dby_real[i][1]:.2e}A, Phase: {dby_real[i][2]:.2f}deg, GBW: {dby_real[i][3]:.2e}Hz"
        data_list.append(f"X: [{x_str}] | Y: [{y_str}]")

    return "\n".join(data_list)


def llm_ranking_actual(dbx: torch.Tensor, dby_real: List[List[float]], param_names: list,
                       logger: logging.Logger) -> list:
    """
    Calls the LLM API (OpenAI compatible) to get the LLM-guided parameter ranking (X_LLM)
    using structured JSON output.
    """

    # --- 1. Define JSON Schema for Structured Output ---
    # NOTE: Using a simple text-only schema for compatibility with most API wrappers.
    # The prompt forces JSON output.
    ranking_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "parameter": {"type": "STRING",
                              "description": "The name of the parameter from the input list (e.g., L3, W4)."},
                "score": {"type": "NUMBER",
                          "description": "A normalized importance score between 0 and 100, where 100 is most critical."}
            },
            "required": ["parameter", "score"],
            "propertyOrdering": ["parameter", "score"]
        },
        "description": "A ranked list of all 12 circuit parameters by their importance to achieving the goal."
    }

    if not OPENAI_API_KEY or not BASE_URL:
        logger.error("LLM API key or Base URL not found in environment. Cannot perform LLM ranking.")
        # Fallback to neutral ranking
        llm_sorted_params_fallback = [(name, 50.0) for name in param_names]
        return [llm_sorted_params_fallback[i:i + 4] for i in range(0, 12, 4)]

    # --- 2. Construct Prompt and Payload ---
    data_sample = format_data_for_prompt(dbx, dby_real, param_names)

# You may modify the Prompt
    system_prompt = (
        "You are an expert analog circuit designer and optimization agent. Your task is to rank the criticality of 12 "
        "transistor sizing parameters for a Two-Stage Op-Amp. The design goal is to MINIMIZE DC Current (I) while meeting "
        "constraints (Gain > 60dB, PM > 60deg, GBW > 4MHz). "
        "Analyze the list of parameters, their values, and the provided simulation data to assign an importance score "
        "for each parameter based on its influence on the required performance metrics (especially minimizing Current). "
        "Your response MUST be a single, strict JSON array of objects, as described in the schema. Do not include any other text."
    )

    user_query = (
        f"Circuit Parameters: {', '.join(param_names)}\n\n"
        "Optimization Goal: Minimize DC Current while meeting all other constraints.\n\n"
        "Simulation Data Samples (X: Input Parameters, Y: Output Metrics):\n"
        f"---BEGIN DATA---\n{data_sample}\n---END DATA---\n\n"
        "Please provide the ranking for all 12 parameters in the requested JSON format, ranked by score (descending)."
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "response_format": {"type": "json_object"},
        "seed": 42  # For deterministic reasoning
    }

    # --- 3. Make API Call (with exponential backoff) ---
    logger.info("Calling LLM API (%s) for parameter ranking...", LLM_MODEL)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                LLM_API_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                },
                data=json.dumps(payload)
            )
            response.raise_for_status()

            result = response.json()

            # --- 4. Parse Response and Format Output ---
            # Extract JSON from the content field
            json_text = result['choices'][0]['message']['content']
            llm_ranking_data = json.loads(json_text)

            # The JSON object should contain the 'ranking' array if the model followed the schema
            if isinstance(llm_ranking_data, dict) and 'ranking' in llm_ranking_data:
                llm_ranking_data = llm_ranking_data['ranking']
            elif not isinstance(llm_ranking_data, list):
                # Handle cases where the model returns JSON but not an array/expected structure
                raise ValueError("LLM response is not a direct JSON array or does not contain 'ranking' key.")

            # Convert JSON data into the required format: list of tuples (param_name, score)
            llm_sorted_params = []
            for item in llm_ranking_data:
                if item['parameter'] in param_names:
                    llm_sorted_params.append((item['parameter'], item['score']))

            # Ensure all 12 parameters are present (critical for grouping)
            if len(llm_sorted_params) < len(param_names):
                logger.warning("LLM only returned %d parameters. Filling remaining with neutral score.",
                               len(llm_sorted_params))
                present_names = {p[0] for p in llm_sorted_params}
                for name in param_names:
                    if name not in present_names:
                        llm_sorted_params.append((name, 0.0))  # Add with minimum score

            # Sort and group
            llm_sorted_params = sorted(llm_sorted_params, key=lambda x: x[1], reverse=True)
            groups_llm = [llm_sorted_params[i:i + 4] for i in range(0, 12, 4)]

            logger.info("LLM ranking successfully received and parsed.")
            return groups_llm

        except requests.exceptions.RequestException as e:
            logger.warning("API call attempt %d/%d failed: %s. Retrying in %ds...", attempt + 1, MAX_RETRIES, e,
                           2 ** attempt)
            time.sleep(2 ** attempt)
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse LLM response structure: %s. Raw response: %s", e, response.text)
            break

    logger.error("LLM parameter ranking failed after all retries. Falling back to neutral MI grouping.")
    # Fallback to neutral ranking
    llm_sorted_params_fallback = [(name, 50.0) for name in param_names]
    groups_llm = [llm_sorted_params_fallback[i:i + 4] for i in range(0, 12, 4)]

    return groups_llm


# --- 2. Dynamic Weight Adjustment Logic  ---

def update_weights(w_old: float, new_valid_y: List[List[float]], prev_best_y: float,
                   initial_feasible_count: int) -> float:
    # ... (function body remains the same)
    num_better_points = 0
    for y_point in new_valid_y:
        current_i = y_point[1]  # Current is at index 1 in real scale Y
        if current_i < prev_best_y:
            num_better_points += 1

    C_i = initial_feasible_count

    if C_i <= 0:
        return w_old

    update_term = num_better_points / C_i

    return w_old + update_term


def run_focal_optimization(
        initial_x_csv_path: str,  # Path to the CSV of initial feasible designs (X parameters, real space)
        initial_y_csv_path: str,  # Path to the CSV of initial performance results (Y metrics, real space)
        simulation_function: callable,  # The circuit simulation function (e.g., OTA_two_simulation_all)
        task_id: str,
        logger: logging.Logger
) -> Tuple[str, List[float]]:
    """
    Orchestrates the multi-stage Focal Optimization process (ASTRA-FocalOpt Stage 2-4).

    Returns: (output_file_path: str, final_best_result: List[float])
    """

    logger.info("--- FocalOpt (Stage 2) Optimization Task Started ---")
    logger.info(f"Task ID: {task_id}")

    # --- Setup & File Initialization ---
    store_path = "./store"
    os.makedirs(store_path, exist_ok=True)
    SEED = 5
    file_path = os.path.join(store_path, f"focalopt_{task_id}_SEED_{SEED}.csv")
    csv_file_obj = None
    writer_obj = None
    iter_count_list = [1]  # Global iteration counter

    # --- Global Optimization Weights (ASTRA Core) ---
    w_llm = 0.5
    w_mi = 0.5
    total_stage_iterations = 200  # Total iterations for Stages 2/3 (Stage 4 is 250)

    # Number of feasible points from Stage 1 (used as C_i for weight normalization)
    initial_feasible_count = 0

    try:
        # Open CSV file and write header for streaming results
        csv_file_obj = open(file_path, 'w', newline='', encoding='utf-8')
        writer_obj = csv.writer(csv_file_obj)
        writer_obj.writerow(['iter_times', 'gain(db)', 'dc_current', 'phase', 'GBW(MHZ)'])
        logger.info("FocalOpt results will be streamed to: %s", file_path)

        start_time = time.time()
        seed_set(SEED, logger)
        param_names = ['cap', 'L1', 'L2', 'L3', 'L4', 'L5', 'r', 'W1', 'W2', 'W3', 'W4', 'W5']

        # --- 1. Load Configuration and Initial Data from Stage 1 ---
        param_ranges_full, thresholds = init_OTA_two(logger=logger)

        df_x = pd.read_csv(initial_x_csv_path)
        df_y = pd.read_csv(initial_y_csv_path)

        dbx_tensor_exp = torch.tensor(df_x.values[:, 1:], dtype=torch.double)
        dbx_alter = torch.log(torch.clamp(dbx_tensor_exp, min=1e-12))  # Log space for GP training X
        dby_tensor_real = torch.tensor(df_y.values[:, 1:], dtype=torch.double)

        # Transform Y data to log space for GP training Y
        dby_alter = dby_tensor_real.clone().detach()
        dby_alter[:, 0] = dby_alter[:, 0] / 20.0
        dby_alter[:, 1] = -torch.log(torch.clamp(dby_alter[:, 1], min=1e-12))
        dby_alter[:, 2] = torch.log(torch.clamp(dby_alter[:, 2], min=1e-12))
        dby_alter[:, 3] = torch.log(torch.clamp(dby_alter[:, 3], min=1e-12))

        # Find the Best Feasible Initial Design from the loaded data
        last_valid_x = None
        last_valid_y = None
        best_y_current = float('inf')
        valid_x_list = []
        valid_y_list = []

        for i in range(len(dby_tensor_real)):
            row_y = dby_tensor_real[i]
            if (row_y[0] > thresholds['gain'] and row_y[1] * thresholds['i_multiplier'] < thresholds['i'] and
                    row_y[2] > thresholds['phase'] and row_y[3] > thresholds['gbw']):

                valid_x_list.append(dbx_tensor_exp[i].tolist())
                row_y_list = row_y.tolist()
                valid_y_list.append(row_y_list)
                initial_feasible_count += 1

                if row_y[1] < best_y_current:
                    best_y_current = row_y[1].item()
                    last_valid_x = dbx_tensor_exp[i].tolist()
                    last_valid_y = row_y_list

                    writer_obj.writerow([iter_count_list[0]] + row_y_list)
                    iter_count_list[0] += 1

        if last_valid_x is None:
            logger.warning(
                "No feasible points found in the initial design file. Using the last point as a fallback base.")
            last_valid_x = dbx_tensor_exp[-1].tolist()
            last_valid_y = dby_tensor_real[-1].tolist()
            best_y_current = last_valid_y[1]

        valid_x = valid_x_list
        valid_y = valid_y_list
        all_x = torch.tensor(valid_x, dtype=torch.double)

        logger.info("Starting best current (I) from Stage 1 data: %.3e A", best_y_current)
        logger.info("Total initial feasible points (C_init): %d", initial_feasible_count)

        objective = None
        iter_1 = 50

        bo = BayesianOptimization(
            param_ranges=param_ranges_full,
            n=iter_1, simulation_function=simulation_function, mode='collect_all',
            best_y=best_y_current, dbx_alter=dbx_alter, dby_alter=dby_alter,
            valid_x=valid_x, valid_y=valid_y, last_valid_x=last_valid_x,
            last_valid_y=last_valid_y, thresholds=thresholds, stage='first', logger=logger,
            task_id=task_id, csv_writer=writer_obj, iter_counter=iter_count_list, all_x=all_x
        )

        init_num = 20
        (best_params, best_simulation_result, last_x, last_y, dbx, dby,
         gain_num, I_num_stage1, GBW_num_stage1, phase_num_stage1) = bo.find(init_num=init_num)

        initial_values = last_x
        current_best_y = best_simulation_result[-1][1]  # Update current best I

        # --- 2. Mutual Information (MI) & LLM Ranking Analysis ---
        logger.info("--- FocalOpt Stage 1.5: MI & LLM Ranking Analysis ---")

        # 1. Prepare data for MI and LLM
        FoM_y, FoM_num, gain_num_mi, i_num_mi, phase_num_mi, gbw_num_mi = filter_two_rows(dby, logger=logger)
        dby_for_mi = torch.cat((dby, FoM_y), dim=1)

        # Prepare real Y list for LLM prompt (only valid points)
        llm_input_dbx = dbx[:dby.shape[0]]  # X for all points collected so far (log space)
        llm_input_dby_real = []  # Y in real space for all points collected so far
        for y_log_row in dby:
            y_real, _ = bo.y_revert(y_log_row.unsqueeze(0))
            llm_input_dby_real.append(y_real[0].tolist())

        # 2. MI Ranking (X_MI)
        scores = calculate_scores(
            dbx=dbx, dby=dby_for_mi, I_num=i_num_mi, FoM_num=FoM_num,
            gain_num=gain_num_mi, GBW_num=gbw_num_mi, phase_num=phase_num_mi,
            iter=iter_1 + init_num, init_num=init_num, input_dim=12, logger=logger
        )
        groups_mi = two_sort_and_group(param_names, scores)
        logger.info("MI Ranking Groups: %s", groups_mi)

        # 3. LLM Ranking (X_LLM) - Calling actual LLM API
        groups_llm = llm_ranking_actual(llm_input_dbx, llm_input_dby_real, param_names, logger)
        logger.info("LLM Ranking Groups: %s", groups_llm)

        # --- 3. Stage 2 (Top 4 Parameters) ---
        iter_2 = total_stage_iterations
        param_count = 4

        logger.info("--- FocalOpt Stage 2: Top %d Parameters ---", param_count)

        # Sequentially run LLM-guided BO and MI-guided BO
        for run_idx, (ranking_groups, ranking_name) in enumerate(zip([groups_llm, groups_mi], ["LLM", "MI"])):

            w_current = w_llm if ranking_name == "LLM" else w_mi
            n_iter = int(iter_2 * w_current / (w_llm + w_mi))

            logger.info("--- Stage 2.%d: %s BO (%d iterations, weight=%.2f) ---", run_idx + 1, ranking_name, n_iter,
                        w_current)

            # Select the top 'param_count' parameters from the current ranking list
            params_to_optimize = [g[0] for g in ranking_groups[0]]

            percentage1 = 0.5

            params_and_percentages = [(p, percentage1) for p in params_to_optimize]

            params_indices, param_ranges_stage = get_indices_and_ranges(
                param_names, initial_values, *params_and_percentages
            )

            bo = BayesianOptimization(
                param_ranges=param_ranges_stage, n=n_iter, simulation_function=simulation_function,
                mode='collect_stage', best_y=current_best_y, valid_x=[], valid_y=[], last_valid_x=last_x,
                last_valid_y=last_y, last_all_x=torch.tensor(last_x, dtype=torch.double).unsqueeze(0),
                params_indices=params_indices, thresholds=thresholds, all_x=all_x, logger=logger,
                task_id=task_id, csv_writer=writer_obj, iter_counter=iter_count_list
            )

            stage_init = 20
            (new_best_params, new_valid_y_results, last_x_new, last_y_new, dbx_stage, dby_stage,
             *_) = bo.find(stage_init_num=stage_init, out_dim=4)

            # Update global state
            new_best_points_collected = [res for res in new_valid_y_results if res[1] < current_best_y]

            # Collect all valid points for the new 'all_x' set
            all_x = bo.all_x
            last_x = last_x_new
            last_y = last_y_new

            # Update the current best if an overall better point was found
            if new_best_points_collected:
                current_best_y = min(current_best_y, min(p[1] for p in new_best_points_collected))

            # Update weights (only after a full run to prevent immediate runaway)
            w_llm_old = w_llm
            w_mi_old = w_mi

            if ranking_name == "LLM":
                w_llm = update_weights(w_llm, new_valid_y_results, w_llm_old, initial_feasible_count)
            else:
                w_mi = update_weights(w_mi, new_valid_y_results, w_mi_old, initial_feasible_count)

            # Normalize weights to sum to 1
            total_w = w_llm + w_mi
            w_llm /= total_w
            w_mi /= total_w

            logger.info("Stage 2.%s run complete. New best I: %.3e A. Weights: LLM=%.2f, MI=%.2f", ranking_name,
                        current_best_y, w_llm, w_mi)

        # --- 4. Stage 3 Optimization (Top 8 Parameters) ---
        param_count = 8
        logger.info("--- FocalOpt Stage 3: Top %d Parameters ---", param_count)

        # Select Top 4 from LLM and Top 4 from MI. We use the updated 'last_x' and 'all_x'
        initial_values = last_x  # Use the final best X from Stage 2 as the new center

        for run_idx, (ranking_groups, ranking_name) in enumerate(zip([groups_llm, groups_mi], ["LLM", "MI"])):

            w_current = w_llm if ranking_name == "LLM" else w_mi
            n_iter = int(iter_2 * w_current / (w_llm + w_mi))

            logger.info("--- Stage 3.%d: %s BO (%d iterations, weight=%.2f) ---", run_idx + 1, ranking_name, n_iter,
                        w_current)

            # Parameters: Top 4 (w/ percentage2) + Next 4 (w/ percentage1)
            params_stage_top_4 = [g[0] for g in ranking_groups[0]]
            params_stage_next_4 = [g[0] for g in ranking_groups[1]]

            percentage1 = 0.5
            percentage2 = 0.5

            params_and_percentages = [
                *[(p, percentage2) for p in params_stage_top_4],
                *[(p, percentage1) for p in params_stage_next_4]
            ]

            params_indices, param_ranges_stage = get_indices_and_ranges(
                param_names, initial_values, *params_and_percentages
            )

            bo = BayesianOptimization(
                param_ranges=param_ranges_stage, n=n_iter, simulation_function=simulation_function,
                mode='collect_stage', best_y=current_best_y, valid_x=[], valid_y=[], last_valid_x=last_x,
                last_valid_y=last_y, last_all_x=torch.tensor(last_x, dtype=torch.double).unsqueeze(0),
                params_indices=params_indices, thresholds=thresholds, all_x=all_x, logger=logger,
                task_id=task_id, csv_writer=writer_obj, iter_counter=iter_count_list
            )

            (new_best_params, new_valid_y_results, last_x_new, last_y_new, dbx_stage, dby_stage,
             *_) = bo.find(stage_init_num=stage_init, out_dim=4)

            all_x = bo.all_x
            last_x = last_x_new
            last_y = last_y_new

            new_best_points_collected = [res for res in new_valid_y_results if res[1] < current_best_y]

            if new_best_points_collected:
                current_best_y = min(current_best_y, min(p[1] for p in new_best_points_collected))

            # Update weights
            w_llm_old = w_llm
            w_mi_old = w_mi

            if ranking_name == "LLM":
                w_llm = update_weights(w_llm, new_valid_y_results, w_llm_old, initial_feasible_count)
            else:
                w_mi = update_weights(w_mi, new_valid_y_results, w_mi_old, initial_feasible_count)

            total_w = w_llm + w_mi
            w_llm /= total_w
            w_mi /= total_w

            logger.info("Stage 3.%s run complete. New best I: %.3e A. Weights: LLM=%.2f, MI=%.2f", ranking_name,
                        current_best_y, w_llm, w_mi)

        # --- 5. Stage 4 Optimization (All 12 Parameters) ---
        param_count = 12
        iter_4 = 250
        logger.info("--- FocalOpt Stage 4: All %d Parameters (Final Stage) ---", param_count)

        # Select All 12 parameters (Top 4, Next 4, Last 4) from the ranking.
        initial_values = last_x

        for run_idx, (ranking_groups, ranking_name) in enumerate(zip([groups_llm, groups_mi], ["LLM", "MI"])):

            w_current = w_llm if ranking_name == "LLM" else w_mi
            # Use total iter_4 budget
            n_iter = int(iter_4 * w_current / (w_llm + w_mi))

            logger.info("--- Stage 4.%d: %s BO (%d iterations, weight=%.2f) ---", run_idx + 1, ranking_name, n_iter,
                        w_current)

            # Parameters: Top 4 (w/ percentage2) + Next 4 (w/ percentage2) + Last 4 (w/ percentage1)
            params_stage_top_4 = [g[0] for g in ranking_groups[0]]
            params_stage_next_4 = [g[0] for g in ranking_groups[1]]
            params_stage_last_4 = [g[0] for g in ranking_groups[2]]

            percentage1 = 0.5
            percentage2 = 0.5

            params_and_percentages = [
                *[(p, percentage2) for p in params_stage_top_4],
                *[(p, percentage2) for p in params_stage_next_4],
                *[(p, percentage1) for p in params_stage_last_4]
            ]

            params_indices, param_ranges_stage = get_indices_and_ranges(
                param_names, initial_values, *params_and_percentages
            )

            bo = BayesianOptimization(
                param_ranges=param_ranges_stage, n=n_iter, simulation_function=simulation_function,
                mode='collect_stage', best_y=current_best_y, valid_x=[], valid_y=[], last_valid_x=last_x,
                last_valid_y=last_y, last_all_x=torch.tensor(last_x, dtype=torch.double).unsqueeze(0),
                params_indices=params_indices, thresholds=thresholds, all_x=all_x,
                stage='last', logger=logger, task_id=task_id, csv_writer=writer_obj, iter_counter=iter_count_list
            )

            (new_best_params, new_valid_y_results, last_x_new, last_y_new, dbx_stage, dby_stage,
             *_) = bo.find(stage_init_num=stage_init, out_dim=4)

            # No update to all_x in the final stage is critical for correctness, so we omit that logic.
            last_x = last_x_new
            last_y = last_y_new

            new_best_points_collected = [res for res in new_valid_y_results if res[1] < current_best_y]

            if new_best_points_collected:
                current_best_y = min(current_best_y, min(p[1] for p in new_best_points_collected))

            # Weights are not updated after the final stage, but we track performance.

            logger.info("Stage 4.%s run complete. New best I: %.3e A. Weights: LLM=%.2f, MI=%.2f", ranking_name,
                        current_best_y, w_llm, w_mi)

        # --- 6. Finalization ---
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Total FocalOpt execution time: %.4f seconds", execution_time)
        logger.info("--- FocalOpt (Stage 2) Optimization Task Success ---")

        if csv_file_obj:
            csv_file_obj.close()

        # Return the final best result found
        final_best = last_y
        return file_path, final_best


    except Exception as e:
        logger.error("FocalOpt Task Failed: %s", e)
        logger.error(traceback.format_exc())

        if csv_file_obj:
            try:
                csv_file_obj.close()
            except:
                pass

        raise e
