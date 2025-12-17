import numpy as np

PH_MIN = 7.0
PH_MAX = 7.5

# Tunable constants (can be calibrated later)
NAOH_PER_PH = 0.5     # NaOH units per pH
NAOH_PER_PAC = 0.1    # NaOH units per PAC


def estimate_ph(ph1, pac, naoh):
    """
    Linearized pH estimate near neutral.
    Directionally correct and stable.
    """
    return ph1 - 0.1 * pac + 0.1 * naoh


def adjust_naoh(naoh_base, ph1, pac, ph_pred):
    """
    Adjust NAOH only if estimated pH violates limits.
    """
    naoh = naoh_base

    # compensate acidity introduced by PAC
    naoh += NAOH_PER_PAC * pac

    # pH band enforcement
    if ph_pred < PH_MIN:
        naoh += (PH_MIN - ph_pred) * NAOH_PER_PH

    elif ph_pred > PH_MAX:
        naoh -= (ph_pred - PH_MAX) * NAOH_PER_PH

    return max(0.0, naoh)
