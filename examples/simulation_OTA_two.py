import math
import os
import statistics
# Assuming KATO structure is available, keep this path as is if required by the environment
from KATO.Data.lyngspice_master.lyngspice_master.lyngspice.lyngspice import NgSpice
import numpy as np
import torch
import time
import shutil
import pandas as pd
from scipy.interpolate import interp1d

# Prevent OpenMP runtime errors on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))

# Netlist file paths (assuming they are in a 'netlists' folder relative to the script)
file_path_OTA_two_gmid_new = os.path.join(current_dir, "netlists", "ICCAD_OTA_two_new.cir")
file_path_OTA_two_all = os.path.join(current_dir, 'netlists', 're_OTA_two_all_netlist.cir')

# Base directory for Look-Up Tables (LUTs)
GMID_LUT_DIR = os.path.join(current_dir, "gmid_LUT")


def find_closest_points_indices(lst, aim_L):
    """Finds the indices of the two points in lst closest to aim_L (one above, one below)."""
    above_1_index = None
    below_1_index = None

    for i, num in enumerate(lst):
        if num <= aim_L and (above_1_index is None or aim_L >= lst[above_1_index]):
            above_1_index = i
            if i + 1 == len(lst):
                below_1_index = i
                break
            if lst[i + 1] > aim_L and below_1_index is None:
                below_1_index = i + 1
                break

    if above_1_index is None:
        if below_1_index != 0:
            above_1_index = below_1_index - 1
        else:
            above_1_index = below_1_index + 1
    if below_1_index is None:
        if above_1_index != 0:
            below_1_index = above_1_index - 1
        else:
            below_1_index = above_1_index + 1

    return below_1_index, above_1_index


def calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L):
    """Performs linear interpolation to find ID/W corresponding to aim_L."""
    # Linear interpolation: L = k * (ID/W) + b
    k = (L_below - L_above) / (idoverw_below - idoverw_above)
    b = L_below - k * idoverw_below
    # Solve for ID/W when L = aim_L
    return (aim_L - b) / k


def calculate_w_linear_NMOS_pro(aim_L, aim_I, gmid):
    """Calculates NMOS width based on L, ID, and GM/ID using linear interpolation on LUTs."""
    # Read CSV file (Updated: use relative path)
    lut_path = os.path.join(GMID_LUT_DIR, f'nmos_gmid{int(gmid)}.csv')
    try:
        df = pd.read_csv(lut_path)
    except FileNotFoundError:
        print(f"Error: NMOS LUT file not found at {lut_path}")
        return 1e-9  # Return minimum width as fallback

    # Extract L and ID/W values
    L_values = df[f'L (GM/ID=ID/W (GM/ID={int(gmid)}))'].values
    idoverw_values = df['ID/W'].values

    if len(L_values) < 2:
        return aim_I / 1e-6  # Safety fallback

    below_1_index, above_1_index = find_closest_points_indices(L_values, aim_L)
    L_below, L_above = L_values[below_1_index], L_values[above_1_index]
    idoverw_below, idoverw_above = idoverw_values[below_1_index], idoverw_values[above_1_index]

    result_id_over_w = calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L)

    if result_id_over_w <= 0:
        return 1e-9  # Safety minimum width

    result_w = aim_I / result_id_over_w
    return result_w


def calculate_w_linear_PMOS_pro(aim_L, aim_I, gmid):
    """Calculates PMOS width based on L, ID, and GM/ID using linear interpolation on LUTs."""
    # Read CSV file (Updated: use relative path)
    lut_path = os.path.join(GMID_LUT_DIR, f'pmos_gmid{int(gmid)}.csv')
    try:
        df = pd.read_csv(lut_path)
    except FileNotFoundError:
        print(f"Error: PMOS LUT file not found at {lut_path}")
        return 1e-9  # Return minimum width as fallback

    # Extract L and ID/W values
    L_values = df[f'L (GM/ID=ID/W (GM/ID={int(gmid)}))'].values
    idoverw_values = df['ID/W'].values

    if len(L_values) < 2:
        return aim_I / 1e-6  # Safety fallback

    below_1_index, above_1_index = find_closest_points_indices(L_values, aim_L)
    L_below, L_above = L_values[below_1_index], L_values[above_1_index]
    idoverw_below, idoverw_above = idoverw_values[below_1_index], idoverw_values[above_1_index]

    result_id_over_w = calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L)

    if result_id_over_w <= 0:
        return 1e-9  # Safety minimum width

    result_w = aim_I / result_id_over_w
    return result_w


# This function finds the two points closest to 0dB gain (for GBW/PM calculation)
def find_closest_points_indices_GBW(lst):
    """Finds the indices of the two closest gain points (in dB) to 0 dB, one positive and one negative."""
    # Logic remains the same, assuming 'lst' are dB gain values.
    above_1_index = None
    below_1_index = None
    for i, num in enumerate(lst):
        if num > 0 and (above_1_index is None or 0 < lst[above_1_index]):
            above_1_index = i
            # print(i)
            if i + 1 == len(lst):
                below_1_index = i
                break
            if lst[i + 1] < 0 and below_1_index is None:
                below_1_index = i + 1
                break

    # Handle cases where all points are below 0 or above 0
    if not lst or all(x < 0 for x in lst):
        return len(lst) - 2, len(lst) - 1 if len(lst) >= 2 else (0, 0)
    if all(x > 0 for x in lst):
        return 0, 1 if len(lst) >= 2 else (0, 0)

    # Simplified logic for finding indices when the crossover is near
    if above_1_index is None or below_1_index is None:
        if lst[0] > 0 and lst[-1] < 0:
            for i in range(len(lst) - 1):
                if lst[i] > 0 and lst[i + 1] < 0:
                    above_1_index = i
                    below_1_index = i + 1
                    break
        elif lst[0] < 0 and lst[-1] < 0:
            above_1_index = len(lst) - 2
            below_1_index = len(lst) - 2

    if above_1_index is None:
        if below_1_index != 0:
            above_1_index = below_1_index - 1
        else:
            above_1_index = below_1_index + 1
    if below_1_index is None:
        if above_1_index != 0:
            below_1_index = above_1_index - 1
        else:
            below_1_index = above_1_index + 1

    # Final check to ensure indices are valid
    if below_1_index >= len(lst) or above_1_index >= len(lst):
        return len(lst) - 1, len(lst) - 1

    return below_1_index, above_1_index


# Calculate average gain for DC gain (Avo)
def calculate_gain(gain):
    """Calculates DC gain by averaging the first 5 gain points (in dB)."""
    return sum(gain[:5]) / 5


# This function is used to calculate Phase Margin
def calculate_phase(gain_below, gain_above, pha_below, pha_above):
    """Performs linear interpolation on log(phase) vs gain to find phase at 0dB."""
    if pha_below < 0:
        return 1
    else:
        # Linear interpolation based on log10(phase)
        k = (gain_above - gain_below) / (pha_above - pha_below)
        b = gain_below - k * pha_below
        # Return log10(phase) when gain is 0 (dB)
        return (0 - b) / k


# This function is used to calculate GBW frequency
def calculate_frequency(gain_below, gain_above, frequency_below, frequency_above):
    """Performs linear interpolation on frequency vs gain to find frequency at 0dB."""
    # Linear interpolation: Gain = k * Frequency + b
    k = (gain_above - gain_below) / (frequency_above - frequency_below)
    b = gain_below - k * frequency_below
    # Solve for Frequency when Gain is 0 (dB)
    return (0 - b) / k


def OTA_two_simulation_gmid_pro(x, gmid1, gmid2, gmid3, gmid4, gmid5):
    """
    Simulates the two-stage OTA, calculating W using gmid LUTs.

    :param x: Tensor containing C_comp, k1, k2, L1-L5, R.
    :param gmid1-gmid5: Desired GM/ID ratios (integers 2-25).
    :return: Tensor [log10(Av), log(Idc), log(PM), log(GBW)]
    """
    n = x.size(0)
    x = x.numpy()
    results = []

    for i in range(n):
        cap = x[i][0].item()
        k1 = x[i][1].item()
        k2 = x[i][2].item()
        l1 = x[i][3].item()
        l2 = x[i][4].item()
        l3 = x[i][5].item()
        l4 = x[i][6].item()
        l5 = x[i][7].item()
        r = x[i][8].item()

        ng = NgSpice()  # Initialize NgSpice
        print(f'------------------re_OTA_two_new-------------------')

        # Rewrite netlist with calculated W values
        new_file_path = write_data_OTA_two_gmid_pro(
            file_path_OTA_two_gmid_new, cap=cap, k1=k1, k2=k2, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5,
            r=r, gmid1=gmid1, gmid2=gmid2, gmid3=gmid3, gmid4=gmid4, gmid5=gmid5
        )

        data, units = ng.run(new_file_path)  # Simulate

        try:
            # AC analysis data
            voltage = data['ac1']['net4'] / data['ac1']['net3']
            frequency = data['ac1']['frequency']

            # DC current (Power proxy) - based on v0#branch current
            # Note: The ' - 8e-5' seems specific to this circuit's bias current setting.
            dc_current = np.abs(data['op1']['v0#branch']) - 8e-5

            voltage_abs = np.abs(voltage)
            voltage_pha = np.degrees(np.angle(voltage))

            # Calculate Gain (dB)
            gain = [20 * math.log10(a) for a in voltage_abs]
            result_gain_db = calculate_gain(gain)

            # Find the crossover point (0dB)
            below_1_index, above_1_index = find_closest_points_indices_GBW(gain)

            GBW = 0
            phase = 0

            # Check for valid crossover
            if below_1_index == above_1_index or result_gain_db < 0:
                GBW = 0
                phase = 0
            else:
                a, b = gain[below_1_index], gain[above_1_index]

                # 1. Calculate GBW
                GBW = calculate_frequency(a, b, frequency[below_1_index], frequency[above_1_index])
                if GBW > 1e7:  # Cap max GBW
                    GBW = 1.0  # Will result in log(1.0) = 0.0
                if GBW < 0:
                    GBW = 1e-6  # Minimum value for log

                # 2. Calculate Phase Margin (PM)
                # PM is calculated using log10(phase) interpolation
                pha_below = voltage_pha[below_1_index]
                pha_above = voltage_pha[above_1_index]

                if pha_below < 0 or pha_above < 0:
                    phase = 1e-6  # Minimum value for log
                else:
                    log_phase_at_0db = calculate_phase(a, b, math.log10(pha_below), math.log10(pha_above))
                    phase = 10 ** log_phase_at_0db
                    if phase > 90:
                        phase = 90.0  # Cap max PM to 90 degrees

            print(f"Input: {x[i]}")
            print(
                f"DC Gain (db)={result_gain_db:.2f}; GBW={GBW:.2e} Hz; I_DC={dc_current[0] * 1e6:.2f} uA; PM={phase:.2f} deg")

            # Final logarithmic transformation for BO
            result_gain_log = result_gain_db / 20  # log10(|Av|)
            dc_current_log = np.log(np.maximum(dc_current, 1e-12))[0]
            phase_log = np.log(np.maximum(phase, 1e-12))
            gbw_log = np.log(np.maximum(GBW, 1e-12))

            results.append([result_gain_log, dc_current_log, phase_log, gbw_log])

            results = torch.tensor(results)
            return results

        except KeyError as e:
            print(f"Simulation failed or key not found: {e}")
            return torch.tensor([[0, 0, 0, 0]])  # Return failure tensor


def OTA_two_simulation_all(x):
    """
    Simulates the two-stage OTA with all L and W provided explicitly.

    :param x: Tensor containing C_comp, L1-L5, R, W1-W5.
    :return: Tensor [log10(Av), log(Idc), log(PM), log(GBW)]
    """
    n = x.size(0)
    # x = x.numpy()
    print(x)
    results = []

    for i in range(n):
        cap = x[i][0].item()
        l1 = x[i][1].item()
        l2 = x[i][2].item()
        l3 = x[i][3].item()
        l4 = x[i][4].item()
        l5 = x[i][5].item()
        r = x[i][6].item()
        w1 = x[i][7].item()
        w2 = x[i][8].item()
        w3 = x[i][9].item()
        w4 = x[i][10].item()
        w5 = x[i][11].item()

        ng = NgSpice()
        print(f'------------------re_OTA_two_all-------------------')

        # Rewrite netlist with all L and W values
        new_file_path = write_data_OTA_two_all(file_path_OTA_two_all, cap=cap, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5,
                                               r=r, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5)  # 重写网表
        data, units = ng.run(new_file_path)

        try:
            # AC analysis data
            voltage = data['ac1']['net4'] / data['ac1']['net3']
            frequency = data['ac1']['frequency']

            # DC current
            dc_current = np.abs(data['op1']['v0#branch']) - 8e-5
            voltage_abs = np.abs(voltage)
            voltage_pha = np.degrees(np.angle(voltage))

            # Calculate Gain (dB)
            gain = [20 * math.log10(a) for a in voltage_abs]
            result_gain_db = calculate_gain(gain)

            # Find the crossover point (0dB)
            below_1_index, above_1_index = find_closest_points_indices_GBW(gain)

            GBW = 0
            phase = 0

            # Check for valid crossover
            if below_1_index == above_1_index or result_gain_db < 0:
                GBW = 0
                phase = 0
            else:
                a, b = gain[below_1_index], gain[above_1_index]

                # 1. Calculate GBW
                GBW = calculate_frequency(a, b, frequency[below_1_index], frequency[above_1_index])
                if GBW > 1e7:
                    GBW = 1.0
                if GBW < 0:
                    GBW = 1e-6

                # 2. Calculate Phase Margin (PM)
                pha_below = voltage_pha[below_1_index]
                pha_above = voltage_pha[above_1_index]

                if pha_below < 0 or pha_above < 0:
                    phase = 1e-6
                else:
                    log_phase_at_0db = calculate_phase(a, b, math.log10(pha_below), math.log10(pha_above))
                    phase = 10 ** log_phase_at_0db
                    if phase > 90:
                        phase = 90.0

            print(f"Input: {x[i]}")
            print(
                f"DC Gain (db)={result_gain_db:.2f}; GBW={GBW:.2e} Hz; I_DC={dc_current[0] * 1e6:.2f} uA; PM={phase:.2f} deg")

            # Final logarithmic transformation for BO
            result_gain_log = result_gain_db / 20
            dc_current_log = np.log(np.maximum(dc_current, 1e-12))[0]
            phase_log = np.log(np.maximum(phase, 1e-12))
            gbw_log = np.log(np.maximum(GBW, 1e-12))

            results.append([result_gain_log, dc_current_log, phase_log, gbw_log])

            results = torch.tensor(results)
            return results

        except KeyError as e:
            print(f"Simulation failed or key not found: {e}")
            return torch.tensor([[0, 0, 0, 0]])


def write_data_OTA_two_gmid_pro(filename, cap=3e-12, k1=1, k2=8, l1=2e-6, l2=2e-6, l3=2e-6, l4=1e-6, l5=1e-6,
                                r=3000, gmid1=0, gmid2=0, gmid3=0, gmid4=0, gmid5=0):
    """
    Rewrites the netlist file by calculating W based on L and GM/ID,
    and inserting all parameters.
    """
    src_file = filename
    # Updated: use a temporary file path in the current directory
    new_file = os.path.join(current_dir, 'temp_ICCAD_OTA_two_new2.cir')
    shutil.copy(src_file, new_file)

    # Open file and read content
    with open(new_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    # 1. Calculate W based on L, ID, and GM/ID
    # ID is derived from the total bias current (implicitly 40uA) and k ratios
    W_MIN = 180e-9  # Minimum allowed width

    w1 = calculate_w_linear_NMOS_pro(l1, 20e-6 * k1, gmid1)
    if w1 < W_MIN:
        w1 = 1e-9
    w2 = calculate_w_linear_PMOS_pro(l2, 40e-6 * k2, gmid2)
    if w2 < W_MIN:
        w2 = 1e-9
    w3 = calculate_w_linear_PMOS_pro(l3, 20e-6 * k1, gmid3)
    if w3 < W_MIN:
        w3 = 1e-9
    w4 = calculate_w_linear_NMOS_pro(l4, 40e-6 * k1, gmid4)
    if w4 < W_MIN:
        w4 = 1e-9
    w5 = calculate_w_linear_NMOS_pro(l5, 40e-6 * k2, gmid5)
    if w5 < W_MIN:
        w5 = 1e-9

    # 2. Format parameters for netlist (using u and p suffixes)
    cap_f = format(cap * 1e12, ".1f")
    l1_f = format(l1 * 1e6, ".2f")
    l2_f = format(l2 * 1e6, ".2f")
    l3_f = format(l3 * 1e6, ".2f")
    l4_f = format(l4 * 1e6, ".2f")
    l5_f = format(l5 * 1e6, ".2f")
    w1_f = format(w1 * 1e6, ".2f")
    w2_f = format(w2 * 1e6, ".2f")
    w3_f = format(w3 * 1e6, ".2f")
    w4_f = format(w4 * 1e6, ".2f")
    w5_f = format(w5 * 1e6, ".2f")

    print(f"Param values: C={cap_f}pf, L1-L5={l1_f}u-{l5_f}u, R={r}, W1-W5={w1_f}u-{w5_f}u")

    # 3. Modify the .PARAM line
    modified_lines = []
    param_line_new = (
        f".PARAM cap={cap_f}pf l1={l1_f}u l2={l2_f}u l3={l3_f}u l4={l4_f}u l5={l5_f}u r={r} "
        f"w1={w1_f}u w2={w2_f}u w3={w3_f}u w4={w4_f}u w5={w5_f}u \n"
    )

    param_found = False
    for line in lines:
        if line.strip().startswith(".PARAM"):
            modified_lines.append(param_line_new)
            modified_lines.append("\n")
            param_found = True
        elif not param_found and line.strip() == "":  # Avoid adding a new line if param is not found yet
            modified_lines.append(line)
        elif param_found:
            modified_lines.append(line)
        else:
            modified_lines.append(line)

    if not param_found:
        modified_lines.insert(0, param_line_new)

    # Write the modified content back to the new file
    with open(new_file, 'w', encoding="utf-8") as file:
        file.writelines(modified_lines)

    return new_file


def write_data_OTA_two_all(filename, cap=3e-12, l1=8e-7, l2=8e-7, l3=8e-7, l4=1e-6, l5=1e-6, r=3000, w1=1.32e-6,
                           w2=4.8e-5, w3=6.6e-6, w4=3.3e-6, w5=1.2e-5):
    """
    Rewrites the netlist file by inserting all L and W parameters explicitly.
    """
    src_file = filename
    # Updated: use a temporary file path in the current directory
    new_file = os.path.join(current_dir, 'temp_retest_OTA_two_all_netlist_new.cir')
    shutil.copy(src_file, new_file)

    # Open file and read content
    with open(new_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    # Format parameters for netlist
    cap_f = format(cap * 1e12, ".1f")
    l1_f = format(l1 * 1e6, ".2f")
    l2_f = format(l2 * 1e6, ".2f")
    l3_f = format(l3 * 1e6, ".2f")
    l4_f = format(l4 * 1e6, ".2f")
    l5_f = format(l5 * 1e6, ".2f")
    w1_f = format(w1 * 1e6, ".2f")
    w2_f = format(w2 * 1e6, ".2f")
    w3_f = format(w3 * 1e6, ".2f")
    w4_f = format(w4 * 1e6, ".2f")
    w5_f = format(w5 * 1e6, ".2f")

    print(f"Param values: C={cap_f}pf, L1-L5={l1_f}u-{l5_f}u, R={r}, W1-W5={w1_f}u-{w5_f}u")

    # Modify the .PARAM line
    modified_lines = []
    param_line_new = (
        f".PARAM cap={cap_f}pf l1={l1_f}u l2={l2_f}u l3={l3_f}u l4={l4_f}u l5={l5_f}u r={r} "
        f"w1={w1_f}u w2={w2_f}u w3={w3_f}u w4={w4_f}u w5={w5_f}u\n"
    )

    param_found = False
    for line in lines:
        if line.strip().startswith(".PARAM"):
            modified_lines.append(param_line_new)
            modified_lines.append("\n")
            param_found = True
        elif not param_found and line.strip() == "":
            modified_lines.append(line)
        elif param_found:
            modified_lines.append(line)
        else:
            modified_lines.append(line)

    if not param_found:
        modified_lines.insert(0, param_line_new)

    # Write the modified content back to the new file
    with open(new_file, 'w', encoding="utf-8") as file:
        file.writelines(modified_lines)

    return new_file


if __name__ == "__main__":
    # Test cases (inputs for simulation)
    # x = [C_comp, k1, k2, L1, L2, L3, L4, L5, R]
    x21 = torch.tensor([[3e-12, 0.55, 2, 8e-7, 8e-7, 8e-7, 1e-6, 1e-6, 3000]])

    # Test GM/ID simulation with all GM/ID = 10
    print(OTA_two_simulation_gmid_pro(x21, gmid1=10, gmid2=10, gmid3=10, gmid4=10, gmid5=10))

    # Example of full-sizing simulation (requires all W values in the tensor)
    # x3 = torch.tensor([[4.9342e-11, 3.9489e-06, 2.1800e-06, 1.5304e-06, 9.4462e-07, 1.1512e-06,
    #      7.8105e+03, 2.5833e-05, 5.5664e-04, 1.0438e-05, 4.3005e-06, 1.5804e-05]])
    # print(OTA_two_simulation_all(x3))
