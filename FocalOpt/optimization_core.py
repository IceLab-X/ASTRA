import random
import time
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedObjective
import copy
import csv
import logging
from typing import List, Dict, Any, Tuple
import os
# Imports from modularized files
from .utility_functions import ota_two_fom_cal, get_indices_and_ranges, set_log_bounds, two_sort_and_group, seed_set
from .mi_analysis import filter_two_rows

# Parameter precision setting
torch.set_default_dtype(torch.double)


class BayesianOptimization():
    def __init__(self, param_ranges: list, simulation_function: callable, mode: str = 'find one',
                 n: int = None, best_y: float = 3.6e-4, stage: str = 'none',
                 dbx_alter: torch.Tensor = None, dby_alter: torch.Tensor = None,
                 valid_x: list = None, valid_y: list = None, last_valid_x: list = None,
                 last_valid_y: list = None, last_all_x: torch.Tensor = None,
                 params_indices: list = None, thresholds: dict = None, all_x: torch.Tensor = None,
                 fom_flag: int = 0, min_max_list: list = None, inputdim: int = 12,
                 logger: logging.Logger = None, task_id: str = "default_task",
                 csv_writer: csv.writer = None, iter_counter: List[int] = None):

        self.param_ranges = param_ranges
        self.simulation_function = simulation_function
        self.n = n  # Number of optimization iterations
        self.mode = mode  # Run mode ('collect_all', 'collect_stage', 'find_one', etc.)
        self.best_y = best_y  # Best current performance metric (usually min current I)
        self.stage = stage  # Stage identifier ('first', 'last')
        self.dbx_alter = dbx_alter  # X values for GP training (log space)
        self.dby_alter = dby_alter  # Y values for GP training (raw/transformed log space)
        self.valid_x = valid_x  # Stores the overall best valid X designs (real space)
        self.valid_y = valid_y  # Stores the overall best valid Y results (real space)
        self.last_valid_x = last_valid_x  # Stores the best valid X found so far (real space)
        self.last_valid_y = last_valid_y  # Stores the best valid Y found so far (real space)
        self.last_all_x = last_all_x  # Full 12-dim X of the last valid point (real space, used in stage mode)
        self.params_indices = params_indices  # Indices of the parameters currently being optimized (unbound)
        self.thresholds = thresholds  # Constraint thresholds
        self.all_x = all_x  # Stores all feasible X points (real space, used in stage mode)
        self.fom_flag = fom_flag  # Flag for unconstrained optimization (0=constrained, 1=unconstrained FoM)
        self.min_max_list = min_max_list  # Min/Max ranges for FoM calculation
        self.inputdim = inputdim  # Total input dimension (12 for OTA2)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.task_id = task_id
        self.csv_writer = csv_writer  # CSV stream writer object
        self.iter_counter = iter_counter  # Global iteration counter [int] list

        # Initialize feasibility counters
        self.gain_num = 1
        self.I_num = 1
        self.GBW_num = 1
        self.phase_num = 1

        # Bandgap flags removed for modularity of OTA2 focus
        self.bandgap_flag = 0

    @staticmethod
    def y_revert(y_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverts the GP training log-transformed Y values back to real scale (test_y)
        and prepares the raw log values (y_value) for concatenation by minimizing I.
        """
        # Numerical reversion to real scale (test_y)
        test_y = torch.empty(1, 4)
        test_y[0][0] = 20 * y_value[0][0]  # Gain (linear -> dB)
        test_y[0][1:] = torch.exp(y_value[0][1:])  # Current, Phase, GBW (log -> exp)

        # Invert the current value (index 1) in the log-transformed tensor
        # to convert the problem from minimization (current) to maximization (for GP)
        y_value[0][1] = -y_value[0][1]

        return test_y, y_value

    def judge_for_bo(self, test_y: torch.Tensor, new_x: torch.Tensor, thresholds: dict) -> Tuple[int, int]:
        """
        Judgment logic executed during the BO optimization loop.
        Checks constraints and updates the overall best point.

        Returns: (cat_flag: int, next_flag: int)
            cat_flag: 1 (feasible and accepted), 0 (infeasible), -2 (sim failure/PM<2)
            next_flag: 0 (new best found), 1 (no new best found), determines stage termination
        """
        if self.fom_flag == 0:
            # Constrained Optimization Mode

            # Check for simulation failures (NaN/Inf)
            if torch.isnan(test_y).any() or torch.isinf(test_y).any() or test_y[:, 2] < 2:
                self.logger.warning("Simulation result contains NaN/Inf or PM < 2, skipping point.")
                return -2, 0

                # Check if all constraints are met (Feasible Region)
            if (test_y[:, 0] > thresholds['gain'] and
                    test_y[:, 1] * thresholds['i_multiplier'] < thresholds['i'] and
                    test_y[:, 2] > thresholds['phase'] and
                    test_y[:, 3] > thresholds['gbw']):

                # Check if it is better than existing data (lower current I)
                is_better = test_y[0][1].item() < self.best_y

                if is_better or self.mode.startswith('collect'):

                    if is_better:
                        # NEW GLOBAL BEST FOUND
                        self.best_y = test_y[0][1].item()
                        new_best_list = test_y[0].tolist()
                        self.valid_x.append(new_x[0].tolist())
                        self.valid_y.append(new_best_list)
                        self.last_valid_x = new_x[0].tolist()
                        self.last_valid_y = new_best_list

                        # Stream write to CSV
                        if self.csv_writer and self.iter_counter is not None:
                            try:
                                row_to_write = [self.iter_counter[0]] + new_best_list
                                self.csv_writer.writerow(row_to_write)
                                self.iter_counter[0] += 1
                            except Exception as e:
                                self.logger.warning(f"CSV streaming write failed: {e}")

                        next_flag = 0  # Reset stage termination counter
                        self.logger.info(f"New best point (Current={self.best_y:.3e}) found.")

                    # Feasible but not better (still accept point for GP training)
                    else:
                        self.valid_x.append(self.last_valid_x)
                        self.valid_y.append(self.last_valid_y)
                        next_flag = 1

                    return 1, next_flag  # Feasible point found

            # If constraints are not met (Infeasible Region)
            else:
                # Inherit the previous best point for plotting/stability
                if self.last_valid_x is not None:
                    self.valid_x.append(self.last_valid_x)
                    self.valid_y.append(self.last_valid_y)

                next_flag = 1
                return 0, next_flag  # Infeasible point found
        else:
            # Unconstrained FoM Mode (Maximization)
            is_better = test_y[0][0].item() > self.best_y
            if is_better:
                self.best_y = test_y[0][0].item()
                self.last_valid_x = new_x[0].tolist()
                self.last_valid_y = test_y[0].tolist()
                next_flag = 0
            else:
                next_flag = 1
            return 1, next_flag  # Always return 1 since FoM is the optimization target

    def judge_for_init(self, test_y: torch.Tensor, i: int, thresholds: dict) -> int:
        """
        Judgment logic executed during the initial random sampling phase.
        Checks constraints and updates the starting best point.

        Returns: True (feasible and accepted), False (infeasible), -2 (sim failure/PM<2)
        """
        # Check for simulation failures (NaN/Inf or PM < 2)
        if torch.isnan(test_y).any() or torch.isinf(test_y).any():
            self.logger.warning(f"Initial point {i + 1} simulation result contains NaN or Inf.")
            return -2
        if test_y[:, 2] < 2:
            return -2

        if self.fom_flag == 0:
            # Constrained Optimization Mode

            # Check if all constraints are met
            if (test_y[:, 0] > thresholds['gain'] and
                    test_y[:, 1] * thresholds['i_multiplier'] < thresholds['i'] and
                    test_y[:, 2] > thresholds['phase'] and
                    test_y[:, 3] > thresholds['gbw']):

                # Check if it is better than existing data (lower current I)
                is_better = test_y[0][1].item() < self.best_y

                # Update data if better
                if is_better:
                    self.best_y = test_y[0][1].item()
                    new_best_list = test_y[0].tolist()
                    self.valid_x.append(torch.exp(self.dbx_alter[i]).tolist())
                    self.valid_y.append(new_best_list)
                    self.last_valid_x = torch.exp(self.dbx_alter[i]).tolist()
                    self.last_valid_y = new_best_list

                    # Stream write to CSV
                    if self.csv_writer and self.iter_counter is not None:
                        try:
                            row_to_write = [self.iter_counter[0]] + new_best_list
                            self.csv_writer.writerow(row_to_write)
                            self.iter_counter[0] += 1
                        except Exception as e:
                            self.logger.warning(f"CSV streaming write failed: {e}")

                # Inherit the previous best value if not better
                else:
                    self.valid_x.append(self.last_valid_x)
                    self.valid_y.append(self.last_valid_y)

                return True  # Point is feasible

            # If constraints are not met
            else:
                # Inherit the previous best value for plotting/stability
                self.valid_x.append(self.last_valid_x)
                self.valid_y.append(self.last_valid_y)
                return False  # Point is infeasible
        else:
            # Unconstrained FoM Mode
            is_better = test_y[0][0].item() > self.best_y
            if is_better:
                self.best_y = test_y[0][0].item()
                self.last_valid_x = torch.exp(self.dbx_alter[i]).tolist()
                self.last_valid_y = test_y[0].tolist()
            else:
                self.valid_x.append(self.last_valid_x)
                self.valid_y.append(self.last_valid_y)
            return True

    @staticmethod
    def save_model(gp: SingleTaskGP, mll: ExactMarginalLogLikelihood, filename: str):
        """Saves the GP model and MLL state dicts."""
        torch.save({
            'model_state_dict': gp.state_dict(),
            'likelihood_state_dict': gp.likelihood.state_dict(),
            'mll_state_dict': mll.state_dict(),
        }, filename)

    @staticmethod
    def load_model(gp: SingleTaskGP, mll: ExactMarginalLogLikelihood, filename: str):
        """Loads the GP model and MLL state dicts."""
        checkpoint = torch.load(filename)
        gp.load_state_dict(checkpoint['model_state_dict'])
        gp.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        mll.load_state_dict(checkpoint['mll_state_dict'])

    def optimize(self):
        """Main Bayesian Optimization loop."""

        # Log bounds for the current stage (used for UCB optimization)
        log_bounds = torch.log(torch.tensor([list(map(lambda x: x[0], self.param_ranges)),
                                             list(map(lambda x: x[1], self.param_ranges))],
                                            dtype=torch.float64))

        flag = 0  # Iteration counter
        end_flag = 0  # Stage termination counter (20 consecutive no-improvements)

        # GP model filename for concurrent safety
        gp_model_filename = f"focalopt_{self.task_id}_gp_model.pth"

        while flag < self.n:
            self.logger.info(f"--------------Starting simulation {flag + 1}/{self.n}-------------")

            # Data preprocessing for GP training (Scaling log-space X to [0, 1])
            self.dbx_alter = self.dbx_alter.double()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(self.dbx_alter)
            x_scaled = torch.tensor(scaled_data, dtype=torch.float64)

            # Transform Y data (dby_alter is log-space) to the FoM for GP training
            if self.fom_flag == 0:
                # Constrained mode: use filtering to generate a FoM for GP
                dby_fom, _, _, _, _, _ = filter_two_rows(self.dby_alter, logger=self.logger)
            else:
                # Unconstrained mode: dby_alter is already the FoM
                dby_fom = self.dby_alter

            # Create GP model
            gp = SingleTaskGP(x_scaled, dby_fom, covar_module=gpytorch.kernels.MaternKernel(nu=2.5))
            gp = gp.to(torch.float64)

            # Add noise constraint for numerical stability
            gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-6))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

            try:
                # Load or train GP model
                if flag == 0 and self.stage == 'last':
                    if os.path.exists(gp_model_filename):
                        BayesianOptimization.load_model(gp, mll, gp_model_filename)
                        self.logger.info(f"Successfully loaded previously saved GP model: {gp_model_filename}")
                    else:
                        self.logger.warning(f"Failed to load GP model: {gp_model_filename} not found. Retraining.")
                        fit_gpytorch_mll(mll)
                else:
                    fit_gpytorch_mll(mll)
            except Exception as e:
                self.logger.error(f"GP model training failed (fit_gpytorch_mll): {e}")
                flag += 1
                continue

            # Acquisition Function parameters (Reset for non-sim-failure iterations)
            num_restarts = 10
            raw_samples = 100
            beta = 2.0

            # Acquisition function (Upper Confidence Bound)
            ucb = UpperConfidenceBound(gp, beta=beta)

            # Candidate point acquisition (Optimizing in scaled [0, 1] space)
            try:
                new_x_scaled, _ = optimize_acqf(
                    acq_function=ucb,
                    bounds=torch.tensor([[0.0] * x_scaled.shape[1], [1.0] * x_scaled.shape[1]], dtype=torch.float64),
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
                # Inverse transform the scaled result back to original log space
                new_x_log_orig_scale = torch.tensor(scaler.inverse_transform(new_x_scaled.detach().numpy()),
                                                    dtype=torch.float64)

            except Exception as e:
                self.logger.error(f"Acquisition function optimization failed (optimize_acqf): {e}")
                flag += 1
                continue

            # Convert acquisition result to real scale for simulation
            new_x = torch.exp(new_x_log_orig_scale.clone().detach())

            # Handle staged optimization: fix unbound parameters, update bound ones
            if self.mode == 'collect_stage':
                if self.all_x is None or self.all_x.size(0) == 0:
                    self.logger.warning("Mode 'collect_stage' but 'all_x' is empty. Using 'last_all_x' as base.")
                    if self.last_all_x is None:
                        self.logger.error("'last_all_x' is also empty, cannot proceed.")
                        flag += 1
                        continue
                    # Create all_x from last_all_x if needed
                    self.all_x = self.last_all_x.clone().detach()

                # Remove duplicate rows from all_x
                self.all_x, _ = torch.unique(self.all_x, dim=0, return_inverse=True)
                # Randomly select a row from feasible points (all_x) to define the fixed parameters
                index = torch.randint(0, self.all_x.size(0), (1,)).item()
                self.last_all_x = self.all_x[index].clone().detach().unsqueeze(0)

                # Update unbound parameters (params_indices) in the 12-dim vector
                for i, index_val in enumerate(self.params_indices):
                    self.last_all_x[0][index_val] = new_x[0][i].item()  # new_x is subset in real scale

                new_y_raw = self.simulation_function(self.last_all_x.clone().detach())
            else:
                # Full unbinding (Stage 1) or find_one mode
                new_y_raw = self.simulation_function(new_x)

            # Post-simulation processing
            test_y, new_y_log = BayesianOptimization.y_revert(new_y_raw)

            # Call judgment function to check constraints and update best point
            cat_flag, next_stage_flag = self.judge_for_bo(test_y, new_x, thresholds=self.thresholds)

            if next_stage_flag == 1 and new_x.shape[1] != self.inputdim:
                end_flag += 1
            else:
                end_flag = 0

            # If feasible in stage mode, add the full 12-dim vector to the set of feasible points
            if cat_flag == 1 and self.mode == 'collect_stage':
                self.all_x = torch.cat([self.all_x, self.last_all_x])

            if cat_flag != -2:
                # Update training set (dbx_alter is log space, dby_alter is log-transformed/FoM)
                self.dbx_alter = torch.cat([self.dbx_alter, new_x_log_orig_scale])
                if self.fom_flag == 0:
                    self.dby_alter = torch.cat([self.dby_alter, new_y_log])
                else:
                    # FoM (unconstrained) is not log-transformed
                    test_y_fom = ota_two_fom_cal(test_y, self.min_max_list)
                    test_y_fom = torch.tensor(test_y_fom).view(1, 1).double()
                    self.dby_alter = torch.cat([self.dby_alter, test_y_fom])
                flag += 1
            else:
                self.logger.warning("Detected simulation failure. Skipping point.")

            # Save GP model on the last round of the initial full unbinding BO
            if flag == self.n and self.stage == 'first':
                BayesianOptimization.save_model(gp, mll, gp_model_filename)
                self.logger.info(f"GP model saved to: {gp_model_filename}")

            # If 20 consecutive no-improvements, terminate stage early
            if end_flag == 20:
                self.logger.info("--------------Stage terminated early (20 consecutive no-improvements)------------")
                return False

        return True

    def find(self, init_num: int = 10, stage_init_num: int = 3, out_dim: int = 4) -> tuple:
        """
        Initialization phase: Performs random sampling to generate initial training data.
        """

        if self.mode == 'collect_all':
            # --- Full Unbinding (Stage 1) Initialization ---

            # Ensure X training data is in log space
            if not isinstance(self.dbx_alter, torch.Tensor):
                self.dbx_alter = torch.tensor(self.dbx_alter, dtype=torch.double)

            # Generate initial random points in log space
            new_x_values_log = torch.empty((init_num, len(self.param_ranges)), dtype=torch.double)
            for i in range(len(self.param_ranges)):
                low, high = self.param_ranges[i]
                log_low, log_high = np.log(low), np.log(high)
                new_x_values_log[:, i] = torch.tensor(np.random.uniform(log_low, log_high, init_num),
                                                      dtype=torch.double)

            # Append new log X values to training set
            self.dbx_alter = torch.cat((self.dbx_alter, new_x_values_log), dim=0)

            # X values in real space for simulation
            new_x_values_sim = torch.exp(new_x_values_log)

            # Ensure Y training data is in log space (from real space CSV data)
            if not isinstance(self.dby_alter, torch.Tensor):
                self.dby_alter = torch.tensor(self.dby_alter, dtype=torch.double)

            # dby_alter is pre-processed in focal_opt_main before being passed here.
            # No further log transform is needed on the initial dby_alter set.

            # --- Sample and Simulate New Points ---
            successful_inits = 0
            init_attempts = 0
            max_init_attempts = init_num * 10

            while successful_inits < init_num and init_attempts < max_init_attempts:
                current_index = successful_inits
                init_attempts += 1

                # Simulate the point in real space
                y_value_real = self.simulation_function(new_x_values_sim[current_index].unsqueeze(0))

                # Convert Y values (test_y: real, y_value_log: log space for GP)
                test_y, y_value_log = BayesianOptimization.y_revert(y_value_real)

                # Unconstrained FoM
                if self.fom_flag == 1:
                    test_y = ota_two_fom_cal(test_y, self.min_max_list)
                    test_y = torch.tensor(test_y).view(1, 1).double()
                    y_value_to_cat = test_y
                else:
                    y_value_to_cat = y_value_log

                # Call judgment function
                # Index 'i' is the index in the full dbx_alter tensor
                init_flag = self.judge_for_init(test_y, self.dby_alter.shape[0] + current_index,
                                                thresholds=self.thresholds)

                if init_flag == -2:
                    self.logger.warning(
                        f"Initial point {current_index + 1} simulation failed (PM < 2 or invalid value), retrying...")
                    # Regenerate random x (log space) for the failed point
                    for t in range(len(self.param_ranges)):
                        low, high = self.param_ranges[t]
                        log_low, log_high = np.log(low), np.log(high)
                        new_x_values_log[current_index, t] = torch.tensor(np.random.uniform(log_low, log_high, 1),
                                                                          dtype=torch.double)

                    # Update the corresponding row in dbx_alter and simulation input
                    self.dbx_alter[self.dby_alter.shape[0] + current_index] = new_x_values_log[current_index]
                    new_x_values_sim[current_index] = torch.exp(new_x_values_log[current_index])
                    continue

                # If simulation successful
                y_value_flattened = y_value_to_cat.flatten().double()
                self.dby_alter = torch.cat((self.dby_alter, y_value_flattened.unsqueeze(0)), dim=0)
                successful_inits += 1

            if successful_inits < init_num:
                self.logger.error(f"Failed to generate enough initial points.")

            # Start Optimization after initialization
            self.optimize()

            # Return results
            return (self.valid_x, self.valid_y, self.last_valid_x, self.last_valid_y,
                    self.dbx_alter, self.dby_alter, self.gain_num, self.I_num,
                    self.GBW_num, self.phase_num)


        elif self.mode == 'collect_stage':
            # --- Staged Optimization Initialization ---

            # Ensure all_x is available (feasible points from previous stages in real space)
            if not isinstance(self.all_x, torch.Tensor):
                self.all_x = torch.tensor(self.all_x, dtype=torch.double)

            # Remove duplicate rows
            self.all_x, _ = torch.unique(self.all_x, dim=0, return_inverse=True)

            if self.all_x.size(0) == 0:
                self.logger.error("In stage mode, 'all_x' is empty, cannot proceed.")
                raise ValueError("'all_x' cannot be empty in stage mode.")

            # Select subset of feasible points from all_x for initialization base
            num_samples = min(self.all_x.size(0), stage_init_num)
            indices = torch.randperm(self.all_x.size(0))[:num_samples] if self.all_x.size(
                0) >= stage_init_num else torch.randint(0, self.all_x.size(0), (stage_init_num,))
            stage_init_x = self.all_x[indices].clone().detach()

            # Initialize empty tensors for training data of the unbound parameters
            num_unbound_params = len(self.param_ranges)
            self.dbx_alter = torch.empty((stage_init_num, num_unbound_params), dtype=torch.double)
            self.dby_alter = torch.empty((0, out_dim), dtype=torch.double)

            # Generate initial random points for the UNBOUND parameters (log space)
            initial_points_log = torch.empty((stage_init_num, num_unbound_params), dtype=torch.double)
            for i in range(num_unbound_params):
                low, high = self.param_ranges[i]
                low_log = np.log(low)
                high_log = np.log(high)
                initial_points_log[:, i] = torch.tensor(np.random.uniform(low_log, high_log, stage_init_num),
                                                        dtype=torch.double)

            self.dbx_alter = initial_points_log  # Log space for training
            initial_points_sim = torch.exp(self.dbx_alter)  # Real space for simulation

            successful_inits = 0
            init_attempts = 0
            max_init_attempts = stage_init_num * 10

            while successful_inits < stage_init_num and init_attempts < max_init_attempts:
                current_index = successful_inits
                init_attempts += 1

                # Construct the full 12-dim real-space vector
                current_stage_x_real = stage_init_x[current_index].clone().detach()
                for i, param_idx in enumerate(self.params_indices):
                    current_stage_x_real[param_idx] = initial_points_sim[current_index][
                        i].item()  # Update unbound params

                # Simulation
                y_value_real = self.simulation_function(current_stage_x_real.unsqueeze(0))

                # Conversion
                test_y, y_value_log = BayesianOptimization.y_revert(y_value_real)

                if self.fom_flag == 1:
                    test_y = ota_two_fom_cal(test_y, self.min_max_list)
                    test_y = torch.tensor(test_y).view(1, 1).double()
                    y_value_to_cat = test_y
                else:
                    y_value_to_cat = y_value_log

                # Judgment
                cat_flag = self.judge_for_init(test_y, successful_inits, thresholds=self.thresholds)

                if cat_flag == -2:
                    self.logger.warning(f"Stage initial point {current_index + 1} simulation failed. Retrying...")
                    # Regenerate random x (log space) for the failed unbound subset
                    for i in range(num_unbound_params):
                        low, high = self.param_ranges[i]
                        low_log, high_log = np.log(low), np.log(high)
                        initial_points_log[current_index, i] = torch.tensor(np.random.uniform(low_log, high_log, 1),
                                                                            dtype=torch.double)

                    self.dbx_alter[current_index] = initial_points_log[current_index]
                    initial_points_sim[current_index] = torch.exp(initial_points_log[current_index])
                    continue

                if cat_flag:  # If feasible, add the full 12-dim vector to the feasible set
                    self.all_x = torch.cat([self.all_x, current_stage_x_real.unsqueeze(0)])

                y_value_flattened = y_value_to_cat.flatten().double()
                self.dby_alter = torch.cat((self.dby_alter, y_value_flattened.unsqueeze(0)), dim=0)
                successful_inits += 1

            # Start Optimization after initialization
            self.optimize()

            # Return results
            return (self.valid_x, self.valid_y, self.last_valid_x, self.last_valid_y,
                    self.dbx_alter, self.dby_alter, self.gain_num, self.I_num,
                    self.GBW_num, self.phase_num, self.last_all_x, self.all_x)

    def print_results(self, dbx: torch.Tensor, dby: list, logger: logging.Logger):
        """Prints the final results of the last valid point."""
        if dbx.nelement() == 0 or not dby:
            logger.warning("No valid data found to print results.")
            return

        # dbx is the final log-space point of the subset/full set
        dbx_np = torch.exp(dbx).numpy()
        dby_np = np.array(dby[-1])  # Last recorded best Y

        logger.info("--- Final Results (print_results) ---")
        logger.info(f"Parameters (Real scale): {dbx_np}")
        logger.info(f"Performance (Real scale): {dby_np}")
        logger.info(f"Best Current I: {self.best_y:.3e} A")

