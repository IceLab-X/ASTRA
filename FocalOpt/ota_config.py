import torch

# Parameter precision setting
torch.set_default_dtype(torch.double)


def set_bounds(value: float) -> list:
    """Calculates parameter bounds based on a 50% window around the initial value."""
    return [value * (1 - 0.5), value * (1 + 0.5)]


def init_OTA_two(logger=None):
    """
    Initializes parameter ranges and constraint thresholds for the OTA Two-Stage Op-Amp.
    The parameters are Cap, L1-L5, R, and W1-W5 (12 parameters total)
    """

    # Saturation region search values (based on typical starting point)
    cap = 4.66e-11
    L1 = 1.52e-6
    L2 = 4.32e-7
    L3 = 1.33e-6
    L4 = 1e-06
    L5 = 1e-06
    r = 9056
    W1 = 1.9608e-5
    W2 = 1.944e-5
    W3 = 8.5785e-5
    W4 = 2.58e-5
    W5 = 9e-6

    # Calculate ranges using a factor of 0.5 (as done in the original config)
    cap_min, cap_max = set_bounds(cap)
    L1_min, L1_max = set_bounds(L1)
    L2_min, L2_max = set_bounds(L2)
    L3_min, L3_max = set_bounds(L3)
    L4_min, L4_max = set_bounds(L4)
    L5_min, L5_max = set_bounds(L5)
    R_min, R_max = set_bounds(r)
    W1_min, W1_max = set_bounds(W1)
    W2_min, W2_max = set_bounds(W2)
    W3_min, W3_max = set_bounds(W3)
    W4_min, W4_max = set_bounds(W4)
    W5_min, W5_max = set_bounds(W5)

    # Define the full 12-parameter range
    param_ranges = [
        (cap_min, cap_max),  # cap (0)
        (L1_min, L1_max),  # L1 (1)
        (L2_min, L2_max),  # L2 (2)
        (L3_min, L3_max),  # L3 (3)
        (L4_min, L4_max),  # L4 (4)
        (L5_min, L5_max),  # L5 (5)
        (R_min, R_max),  # R (6)
        (W1_min, W1_max),  # W1 (7)
        (W2_min, W2_max),  # W2 (8)
        (W3_min, W3_max),  # W3 (9)
        (W4_min, W4_max),  # W4 (10)
        (W5_min, W5_max)  # W5 (11)
    ]

    # Define optimization constraint thresholds
    thresholds = {
        'gain': 60,  # Gain > 60 dB
        'i_multiplier': 1.8,  # I_D * 1.8
        'i': 3e-3,  # DC Current < 3mA (after multiplier)
        'phase': 60,  # Phase Margin (PM) > 60 degrees
        'gbw': 4e6  # Gain Bandwidth Product > 4 MHz
    }

    if logger:
        logger.info("Loaded FocalOpt (OTA Two) parameter ranges and constraints.")

    return param_ranges, thresholds
