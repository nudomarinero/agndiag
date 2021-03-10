import numpy as np


##########################################
# Classification methods
##########################################

# ----------------------#
# Sabater et al. 2012  #
# ----------------------#


def diag_nii_Sabater2012(x, y, c_x, c_y, use_limits=False):
    """
    Diagnostic using the [NII] diagram
    Input:
      x - log([NII]/Halfa)
      y - log([OIII]/Hbeta)
      c_x - detection code for x (0 detection; 1 upper limit; 2 lower limit; 3 non-determined)
      c_y - detection code for y (0 detection; 1 upper limit; 2 lower limit; 3 non-determined)
      use_limits - Take into account the limits if True
    Output:
      diag - Diagnostic code
      c_diag - Code indicating if the diagnostic was applied to an element 1 or 0
    Diagnostic codes:
      1 - SFN
      4 - TO
      5 - NLAGN (Seyfert or LINER)
      8 - TO or SFN
      9 - TO or NLAGN
    """
    x, y, c_x, c_y = numpy_arrays4(x, y, c_x, c_y)
    conditions = [
        [((x >= 0.0) | (y >= 0.8) | ((y - 1.19) * (x - 0.47) <= 0.61)), 5],  # AGN
        [((x < 0.0) & (y < 0.8) & ((y - 1.3) * (x - 0.05) > 0.61)), 1],  # SFN
        [
            (
                (x < 0.0)
                & (y < 0.8)
                & ((y - 1.3) * (x - 0.05) <= 0.61)
                & ((y - 1.19) * (x - 0.47) > 0.61)
            ),
            4,
        ],
    ]  # TO
    if not use_limits:
        cond_init = (c_x == 0) & (c_y == 0)  # Only detections
    else:
        # Take into account only detections for the previous conditions
        conditions = [[c[0] & ((c_x == 0) & (c_y == 0)), c[1]] for c in conditions]
        cond_init = (c_x >= 0) & (c_y >= 0)  # Detections and limits
        conditions_lim = [
            [(x >= 0.0) & ((c_x == 0) | (c_x == 2)), 5],  # AGN right
            [(y >= 0.8) & ((c_y == 0) | (c_y == 2)), 5],  # AGN up
            [
                (
                    ((y - 1.19) * (x - 0.47) <= 0.61)
                    & (
                        ((c_x == 0) & (c_y == 2))
                        | ((c_x == 2) & (c_y == 2))
                        | ((c_x == 2) & (c_y == 0))
                    )
                ),
                5,
            ],  # AGN
            [
                (
                    (
                        (x < 0.0)
                        & (y < 0.8)
                        & ((y - 1.3) * (x - 0.05) <= 0.61)
                        & ((y - 1.19) * (x - 0.47) > 0.61)
                    )
                    & (
                        ((c_x == 0) & (c_y == 2))
                        | ((c_x == 2) & (c_y == 2))
                        | ((c_x == 2) & (c_y == 0))
                    )
                ),
                9,
            ],  # TO or AGN
            [
                (
                    (
                        (x < 0.0)
                        & (y < 0.8)
                        & ((y - 1.3) * (x - 0.05) <= 0.61)
                        & ((y - 1.19) * (x - 0.47) > 0.61)
                    )
                    & (
                        ((c_x == 0) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 0))
                    )
                ),
                8,
            ],  # TO or SFN
            [
                (
                    ((x < 0.0) & (y < 0.8) & ((y - 1.3) * (x - 0.05) > 0.61))
                    & (
                        ((c_x == 0) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 0))
                    )
                ),
                1,
            ],  # SFN
        ]
        # TODO: Add limits for just one line
        conditions.extend(conditions_lim)
    return apply_conditions(cond_init, conditions)


def diag_sii_Sabater2012(x, y, c_x, c_y, use_limits=False):
    """
    Diagnostic using the [SII] diagram
    Input:
      x - log([SII]/Halfa)
      y - log([OIII]/Hbeta)
      c_x - detection code for x (0 detection; 1 upper limit; 2 lower limit)
      c_y - detection code for y (0 detection; 1 upper limit; 2 lower limit)
      use_limits - Take into account the limits if True
    Output:
      diag - Diagnostic code
      c_diag - Code indicating if the diagnostic was applied to an element 1 or 0
    Diagnostic codes:
      1 - SFN
      2 - Seyfert
      3 - LINER
      5 - NLAGN (Seyfert or LINER)
    """
    x, y, c_x, c_y = numpy_arrays4(x, y, c_x, c_y)
    conditions = [
        [
            (
                (((y - 1.3) * (x - 0.32) <= 0.72) | (x >= 0.32))
                & ((1.89 * x - y) <= -0.76)
            ),
            2,
        ],  # Seyfert
        [
            (
                (((y - 1.3) * (x - 0.32) <= 0.72) | (x >= 0.32))
                & ((1.89 * x - y) > -0.76)
            ),
            3,
        ],  # LINER
        [(((y - 1.3) * (x - 0.32) > 0.72) & (x < 0.32)), 1],
    ]  # SFN
    if not use_limits:
        cond_init = (c_x == 0) & (c_y == 0)  # Only detections
    else:
        # Take into account only detections for the previous conditions
        conditions = [[c[0] & ((c_x == 0) & (c_y == 0)), c[1]] for c in conditions]
        cond_init = (c_x >= 0) & (c_y >= 0)  # Detections and limits
        conditions_lim = [
            [
                (((y - 1.3) * (x - 0.32) <= 0.72) & (c_x == 2) & (c_y == 2)),
                5,
            ],  # AGN case 1
            [
                (
                    ((y - 1.3) * (x - 0.32) <= 0.72)
                    & ((1.89 * x - y) <= -0.76)
                    & (c_x == 0)
                    & (c_y == 2)
                ),
                2,
            ],  # Seyfert case 2
            [
                (
                    ((y - 1.3) * (x - 0.32) <= 0.72)
                    & ((1.89 * x - y) <= -0.76)
                    & (c_x == 2)
                    & (c_y == 0)
                ),
                5,
            ],  # AGN case 2
            [
                (
                    ((y - 1.3) * (x - 0.32) <= 0.72)
                    & ((1.89 * x - y) > -0.76)
                    & (c_x == 0)
                    & (c_y == 2)
                ),
                5,
            ],  # AGN case 3
            [
                (
                    ((y - 1.3) * (x - 0.32) <= 0.72)
                    & ((1.89 * x - y) > -0.76)
                    & (c_x == 2)
                    & (c_y == 0)
                ),
                3,
            ],  # LINER case 3
            [
                (
                    (x >= 0.32)
                    & ((1.89 * x - y) <= -0.76)
                    & (((c_x == 2) & (c_y == 1)) | ((c_x == 0) & (c_y == 1)))
                ),
                5,
            ],  # AGN case 4
            [
                (
                    (x >= 0.32)
                    & ((1.89 * x - y) > -0.76)
                    & (((c_x == 2) & (c_y == 1)) | ((c_x == 0) & (c_y == 1)))
                ),
                3,
            ],  # LINER case 5
            [
                (
                    ((y - 1.3) * (x - 0.32) > 0.72)
                    & (
                        ((c_x == 0) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 0))
                    )
                ),
                1,
            ],  # SFN
        ]
        conditions.extend(conditions_lim)
    return apply_conditions(cond_init, conditions)


def diag_oi_Sabater2012(x, y, c_x, c_y, use_limits=False):
    """
    Diagnostic using the [OI] diagram
    Input:
      x - log([OI]/Halfa)
      y - log([OIII]/Hbeta)
      c_x - detection code for x (0 detection; 1 upper limit; 2 lower limit)
      c_y - detection code for y (0 detection; 1 upper limit; 2 lower limit)
      use_limits - Take into account the limits if True
    Output:
      diag - Diagnostic code
      c_diag - Code indicating if the diagnostic was applied to an element 1 or 0
    Diagnostic codes:
      1 - SFN
      2 - Seyfert
      3 - LINER
      5 - NLAGN (Seyfert or LINER)
    """
    x, y, c_x, c_y = numpy_arrays4(x, y, c_x, c_y)
    conditions = [
        [
            (
                (((y - 1.33) * (x + 0.59) <= 0.73) | (x >= -0.59))
                & ((1.18 * x - y) <= -1.3)
            ),
            2,
        ],  # Seyfert
        [
            (
                (((y - 1.33) * (x + 0.59) <= 0.73) | (x >= -0.59))
                & ((1.18 * x - y) > -1.3)
            ),
            3,
        ],  # LINER
        [(((y - 1.33) * (x + 0.59) > 0.73) & (x < -0.59)), 1],
    ]  # SFN
    if not use_limits:
        cond_init = (c_x == 0) & (c_y == 0)  # Only detections
    else:
        # Take into account only detections for the previous conditions
        conditions = [[c[0] & ((c_x == 0) & (c_y == 0)), c[1]] for c in conditions]
        cond_init = (c_x >= 0) & (c_y >= 0)  # Detections and limits
        conditions_lim = [
            [
                (((y - 1.33) * (x + 0.59) <= 0.73) & (c_x == 2) & (c_y == 2)),
                5,
            ],  # AGN case 1
            [
                (
                    ((y - 1.33) * (x + 0.59) <= 0.73)
                    & ((1.18 * x - y) <= -1.3)
                    & (c_x == 0)
                    & (c_y == 2)
                ),
                2,
            ],  # Seyfert case 2
            [
                (
                    ((y - 1.33) * (x + 0.59) <= 0.73)
                    & ((1.18 * x - y) <= -1.3)
                    & (c_x == 2)
                    & (c_y == 0)
                ),
                5,
            ],  # AGN case 2
            [
                (
                    ((y - 1.33) * (x + 0.59) <= 0.73)
                    & ((1.18 * x - y) > -1.3)
                    & (c_x == 0)
                    & (c_y == 2)
                ),
                5,
            ],  # AGN case 3
            [
                (
                    ((y - 1.33) * (x + 0.59) <= 0.73)
                    & ((1.18 * x - y) > -1.3)
                    & (c_x == 2)
                    & (c_y == 0)
                ),
                3,
            ],  # LINER case 3
            [
                (
                    (x >= -0.59)
                    & ((1.18 * x - y) <= -1.3)
                    & (((c_x == 2) & (c_y == 1)) | ((c_x == 0) & (c_y == 1)))
                ),
                5,
            ],  # AGN case 4
            [
                (
                    (x >= -0.59)
                    & ((1.18 * x - y) > -1.3)
                    & (((c_x == 2) & (c_y == 1)) | ((c_x == 0) & (c_y == 1)))
                ),
                3,
            ],  # LINER case 5
            [
                (
                    ((y - 1.33) * (x + 0.59) > 0.73)
                    & (
                        ((c_x == 0) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 1))
                        | ((c_x == 1) & (c_y == 0))
                    )
                ),
                1,
            ],  # SFN
        ]
        conditions.extend(conditions_lim)
    return apply_conditions(cond_init, conditions)


def diag_class_Sabater2012(class_nii, class_sii, class_oi):
    """
    Final classification. Sabater et al. 2012 criteria.
    Codes:
    0 - Unclassified
    1 - SFN
    2 - Seyfert
    3 - LINER
    4 - TO
    5 - NLAGN (Seyfert or LINER)
    8 - TO or SFN
    9 - TO or NLAGN
    10 - Seyfert 1 (not used here)
    """
    sii_oi = np.zeros_like(class_sii)
    s_sii_oi = np.zeros_like(class_sii)
    final = np.zeros_like(class_sii)
    final_to = np.zeros_like(class_sii)
    # Common classification for SII and OI
    # Key for sii oi class: [class sii, class oi, class sii oi]
    table_class = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 2, 2],
        [0, 3, 3],
        [0, 5, 5],
        [1, 0, 1],
        [1, 1, 1],
        [1, 2, 0],
        [1, 3, 0],
        [1, 5, 0],
        [2, 0, 2],
        [2, 1, 0],
        [2, 2, 2],
        [2, 3, 5],
        [2, 5, 2],
        [3, 0, 3],
        [3, 1, 0],
        [3, 2, 5],
        [3, 3, 3],
        [3, 5, 3],
        [5, 0, 5],
        [5, 1, 0],
        [5, 2, 2],
        [5, 3, 3],
        [5, 5, 5],
    ]
    # Key for similarity: [class sii, class oi, similarity sii oi]
    table_sim = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 5, 0],
        [1, 0, 0],
        [1, 1, 1],
        [1, 2, 0],
        [1, 3, 0],
        [1, 5, 0],
        [2, 0, 0],
        [2, 1, 0],
        [2, 2, 1],
        [2, 3, 1],
        [2, 5, 1],
        [3, 0, 0],
        [3, 1, 0],
        [3, 2, 1],
        [3, 3, 1],
        [3, 5, 1],
        [5, 0, 0],
        [5, 1, 0],
        [5, 2, 1],
        [5, 3, 1],
        [5, 5, 1],
    ]
    for t in table_class:
        sii_oi[(class_sii == t[0]) & (class_oi == t[1])] = t[2]
    for s in table_sim:
        s_sii_oi[(class_sii == s[0]) & (class_oi == s[1])] = s[2]
    # Final classification
    # Key for final class: [class nii, class sii oi, similarity sii oi, final class]
    table_class_nii = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 2, 0, 2],
        [0, 3, 0, 3],
        [0, 5, 0, 5],
        [0, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 2, 1, 2],
        [0, 3, 1, 3],
        [0, 5, 1, 5],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 1],
        [1, 3, 0, 1],
        [1, 5, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 3, 1, 1],
        [1, 5, 1, 1],
        [4, 0, 0, 4],
        [4, 1, 0, 4],
        [4, 2, 0, 4],
        [4, 3, 0, 4],
        [4, 5, 0, 4],
        [4, 0, 1, 4],
        [4, 1, 1, 4],
        [4, 2, 1, 4],
        [4, 3, 1, 4],
        [4, 5, 1, 4],
        [5, 0, 0, 5],
        [5, 1, 0, 5],
        [5, 2, 0, 2],
        [5, 3, 0, 3],
        [5, 5, 0, 5],
        [5, 0, 1, 5],
        [5, 1, 1, 1],
        [5, 2, 1, 2],
        [5, 3, 1, 3],
        [5, 5, 1, 5],
        [8, 0, 0, 8],
        [8, 1, 0, 4],
        [8, 2, 0, 4],
        [8, 3, 0, 4],
        [8, 5, 0, 4],
        [8, 0, 1, 8],
        [8, 1, 1, 4],
        [8, 2, 1, 4],
        [8, 3, 1, 4],
        [8, 5, 1, 4],
        [9, 0, 0, 9],
        [9, 1, 0, 4],
        [9, 2, 0, 4],
        [9, 3, 0, 4],
        [9, 5, 0, 4],
        [9, 0, 1, 9],
        [9, 1, 1, 4],
        [9, 2, 1, 4],
        [9, 3, 1, 4],
        [9, 5, 1, 4],
    ]
    for t in table_class_nii:
        final[(class_nii == t[0]) & (sii_oi == t[1]) & (s_sii_oi == t[2])] = t[3]
    # Key for final TO class: [class nii, class sii oi, similarity sii oi, final TO class]
    table_class_to = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 2, 1, 0],
        [0, 3, 1, 0],
        [0, 5, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 2, 0, 0],
        [1, 3, 0, 0],
        [1, 5, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 2, 1, 0],
        [1, 3, 1, 0],
        [1, 5, 1, 0],
        [4, 0, 0, 0],
        [4, 1, 0, 1],
        [4, 2, 0, 2],
        [4, 3, 0, 3],
        [4, 5, 0, 5],
        [4, 0, 1, 0],
        [4, 1, 1, 1],
        [4, 2, 1, 2],
        [4, 3, 1, 3],
        [4, 5, 1, 5],
        [5, 0, 0, 0],
        [5, 1, 0, 0],
        [5, 2, 0, 0],
        [5, 3, 0, 0],
        [5, 5, 0, 0],
        [5, 0, 1, 0],
        [5, 1, 1, 0],
        [5, 2, 1, 0],
        [5, 3, 1, 0],
        [5, 5, 1, 0],
        [8, 0, 0, 0],
        [8, 1, 0, 1],
        [8, 2, 0, 2],
        [8, 3, 0, 3],
        [8, 5, 0, 5],
        [8, 0, 1, 0],
        [8, 1, 1, 1],
        [8, 2, 1, 2],
        [8, 3, 1, 3],
        [8, 5, 1, 5],
        [9, 0, 0, 0],
        [9, 1, 0, 0],
        [9, 2, 0, 2],
        [9, 3, 0, 3],
        [9, 5, 0, 5],
        [9, 0, 1, 0],
        [9, 1, 1, 0],
        [9, 2, 1, 2],
        [9, 3, 1, 3],
        [9, 5, 1, 5],
    ]
    for t in table_class_to:
        final_to[(class_nii == t[0]) & (sii_oi == t[1]) & (s_sii_oi == t[2])] = t[3]
    return final, final_to


def diag_class_OiSiiNii(class_nii, class_sii, class_oi):
    """
    Final classification. Buttiglionne criteria ?
    Codes:
    0 - Unclassified
    1 - SFN
    2 - Seyfert
    3 - LINER
    4 - TO
    5 - NLAGN (Seyfert or LINER)
    8 - TO or SFN
    9 - TO or NLAGN
    10 - Seyfert 1 (not used here)
    """
    final = np.zeros_like(class_sii)
    # Common classification for SII and OI
    # Key for sii oi class: [class sii, class oi, class sii oi]
    table_class = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 2, 0, 2],
        [0, 3, 0, 3],
        [0, 5, 0, 5],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 2, 1, 1],
        [0, 3, 1, 1],
        [0, 5, 1, 1],
        [0, 0, 2, 2],
        [0, 1, 2, 2],
        [0, 2, 2, 2],
        [0, 3, 2, 2],
        [0, 5, 2, 2],
        [0, 0, 3, 3],
        [0, 1, 3, 3],
        [0, 2, 3, 3],
        [0, 3, 3, 3],
        [0, 5, 3, 3],
        [0, 0, 5, 5],
        [0, 1, 5, 5],
        [0, 2, 5, 5],
        [0, 3, 5, 5],
        [0, 5, 5, 5],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 2],
        [1, 3, 0, 3],
        [1, 5, 0, 5],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 3, 1, 1],
        [1, 5, 1, 1],
        [1, 0, 2, 2],
        [1, 1, 2, 2],
        [1, 2, 2, 2],
        [1, 3, 2, 2],
        [1, 5, 2, 2],
        [1, 0, 3, 3],
        [1, 1, 3, 3],
        [1, 2, 3, 3],
        [1, 3, 3, 3],
        [1, 5, 3, 3],
        [1, 0, 5, 5],
        [1, 1, 5, 5],
        [1, 2, 5, 5],
        [1, 3, 5, 5],
        [1, 5, 5, 5],
        [4, 0, 0, 4],
        [4, 1, 0, 1],
        [4, 2, 0, 2],
        [4, 3, 0, 3],
        [4, 5, 0, 5],
        [4, 0, 1, 1],
        [4, 1, 1, 1],
        [4, 2, 1, 1],
        [4, 3, 1, 1],
        [4, 5, 1, 1],
        [4, 0, 2, 2],
        [4, 1, 2, 2],
        [4, 2, 2, 2],
        [4, 3, 2, 2],
        [4, 5, 2, 2],
        [4, 0, 3, 3],
        [4, 1, 3, 3],
        [4, 2, 3, 3],
        [4, 3, 3, 3],
        [4, 5, 3, 3],
        [4, 0, 5, 5],
        [4, 1, 5, 5],
        [4, 2, 5, 5],
        [4, 3, 5, 5],
        [4, 5, 5, 5],
        [5, 0, 0, 5],
        [5, 1, 0, 1],
        [5, 2, 0, 2],
        [5, 3, 0, 3],
        [5, 5, 0, 5],
        [5, 0, 1, 1],
        [5, 1, 1, 1],
        [5, 2, 1, 1],
        [5, 3, 1, 1],
        [5, 5, 1, 1],
        [5, 0, 2, 2],
        [5, 1, 2, 2],
        [5, 2, 2, 2],
        [5, 3, 2, 2],
        [5, 5, 2, 2],
        [5, 0, 3, 3],
        [5, 1, 3, 3],
        [5, 2, 3, 3],
        [5, 3, 3, 3],
        [5, 5, 3, 3],
        [5, 0, 5, 5],
        [5, 1, 5, 5],
        [5, 2, 5, 5],
        [5, 3, 5, 5],
        [5, 5, 5, 5],
        [8, 0, 0, 8],
        [8, 1, 0, 1],
        [8, 2, 0, 2],
        [8, 3, 0, 3],
        [8, 5, 0, 5],
        [8, 0, 1, 1],
        [8, 1, 1, 1],
        [8, 2, 1, 1],
        [8, 3, 1, 1],
        [8, 5, 1, 1],
        [8, 0, 2, 2],
        [8, 1, 2, 2],
        [8, 2, 2, 2],
        [8, 3, 2, 2],
        [8, 5, 2, 2],
        [8, 0, 3, 3],
        [8, 1, 3, 3],
        [8, 2, 3, 3],
        [8, 3, 3, 3],
        [8, 5, 3, 3],
        [8, 0, 5, 5],
        [8, 1, 5, 5],
        [8, 2, 5, 5],
        [8, 3, 5, 5],
        [8, 5, 5, 5],
        [9, 0, 0, 9],
        [9, 1, 0, 1],
        [9, 2, 0, 2],
        [9, 3, 0, 3],
        [9, 5, 0, 5],
        [9, 0, 1, 1],
        [9, 1, 1, 1],
        [9, 2, 1, 1],
        [9, 3, 1, 1],
        [9, 5, 1, 1],
        [9, 0, 2, 2],
        [9, 1, 2, 2],
        [9, 2, 2, 2],
        [9, 3, 2, 2],
        [9, 5, 2, 2],
        [9, 0, 3, 3],
        [9, 1, 3, 3],
        [9, 2, 3, 3],
        [9, 3, 3, 3],
        [9, 5, 3, 3],
        [9, 0, 5, 5],
        [9, 1, 5, 5],
        [9, 2, 5, 5],
        [9, 3, 5, 5],
        [9, 5, 5, 5],
    ]
    for t in table_class:
        final[(class_nii == t[0]) & (class_sii == t[1]) & (class_oi == t[2])] = t[3]
    return final


def diag_class_OiSiiNiiMine(class_nii, class_sii, class_oi):
    """
    Final classification. My criteria in November 2012.
    Codes:
    0 - Unclassified
    1 - SFN
    2 - Seyfert
    3 - LINER
    4 - TO
    5 - NLAGN (Seyfert or LINER)
    8 - TO or SFN
    9 - TO or NLAGN
    10 - Seyfert 1 (not used here)
    """
    final = np.zeros_like(class_sii)
    # Common classification for SII and OI
    # Key for sii oi class: [class sii, class oi, class sii oi]
    table_class = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 2, 0, 2],
        [0, 3, 0, 3],
        [0, 5, 0, 5],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 2, 1, 1],
        [0, 3, 1, 1],
        [0, 5, 1, 1],
        [0, 0, 2, 2],
        [0, 1, 2, 2],
        [0, 2, 2, 2],
        [0, 3, 2, 2],
        [0, 5, 2, 2],
        [0, 0, 3, 3],
        [0, 1, 3, 3],
        [0, 2, 3, 3],
        [0, 3, 3, 3],
        [0, 5, 3, 3],
        [0, 0, 5, 5],
        [0, 1, 5, 5],
        [0, 2, 5, 2],
        [0, 3, 5, 3],
        [0, 5, 5, 5],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 2],
        [1, 3, 0, 3],
        [1, 5, 0, 5],
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 3, 1, 1],
        [1, 5, 1, 1],
        [1, 0, 2, 2],
        [1, 1, 2, 2],
        [1, 2, 2, 2],
        [1, 3, 2, 2],
        [1, 5, 2, 2],
        [1, 0, 3, 3],
        [1, 1, 3, 3],
        [1, 2, 3, 3],
        [1, 3, 3, 3],
        [1, 5, 3, 3],
        [1, 0, 5, 5],
        [1, 1, 5, 5],
        [1, 2, 5, 2],
        [1, 3, 5, 3],
        [1, 5, 5, 5],
        [4, 0, 0, 4],
        [4, 1, 0, 4],
        [4, 2, 0, 2],
        [4, 3, 0, 3],
        [4, 5, 0, 5],
        [4, 0, 1, 4],
        [4, 1, 1, 4],
        [4, 2, 1, 2],
        [4, 3, 1, 3],
        [4, 5, 1, 4],
        [4, 0, 2, 2],
        [4, 1, 2, 2],
        [4, 2, 2, 2],
        [4, 3, 2, 2],
        [4, 5, 2, 2],
        [4, 0, 3, 3],
        [4, 1, 3, 3],
        [4, 2, 3, 3],
        [4, 3, 3, 3],
        [4, 5, 3, 3],
        [4, 0, 5, 5],
        [4, 1, 5, 5],
        [4, 2, 5, 2],
        [4, 3, 5, 5],
        [4, 5, 5, 5],
        [5, 0, 0, 5],
        [5, 1, 0, 1],
        [5, 2, 0, 2],
        [5, 3, 0, 3],
        [5, 5, 0, 5],
        [5, 0, 1, 1],
        [5, 1, 1, 1],
        [5, 2, 1, 1],
        [5, 3, 1, 1],
        [5, 5, 1, 5],
        [5, 0, 2, 2],
        [5, 1, 2, 2],
        [5, 2, 2, 2],
        [5, 3, 2, 2],
        [5, 5, 2, 2],
        [5, 0, 3, 3],
        [5, 1, 3, 3],
        [5, 2, 3, 3],
        [5, 3, 3, 3],
        [5, 5, 3, 3],
        [5, 0, 5, 5],
        [5, 1, 5, 5],
        [5, 2, 5, 2],
        [5, 3, 5, 3],
        [5, 5, 5, 5],
        [8, 0, 0, 8],
        [8, 1, 0, 1],
        [8, 2, 0, 2],
        [8, 3, 0, 3],
        [8, 5, 0, 5],
        [8, 0, 1, 1],
        [8, 1, 1, 1],
        [8, 2, 1, 4],
        [8, 3, 1, 4],
        [8, 5, 1, 4],
        [8, 0, 2, 2],
        [8, 1, 2, 2],
        [8, 2, 2, 2],
        [8, 3, 2, 2],
        [8, 5, 2, 2],
        [8, 0, 3, 3],
        [8, 1, 3, 3],
        [8, 2, 3, 3],
        [8, 3, 3, 3],
        [8, 5, 3, 3],
        [8, 0, 5, 5],
        [8, 1, 5, 5],
        [8, 2, 5, 5],
        [8, 3, 5, 5],
        [8, 5, 5, 5],
        [9, 0, 0, 9],
        [9, 1, 0, 4],
        [9, 2, 0, 2],
        [9, 3, 0, 3],
        [9, 5, 0, 5],
        [9, 0, 1, 1],
        [9, 1, 1, 1],
        [9, 2, 1, 1],
        [9, 3, 1, 1],
        [9, 5, 1, 1],
        [9, 0, 2, 2],
        [9, 1, 2, 2],
        [9, 2, 2, 2],
        [9, 3, 2, 2],
        [9, 5, 2, 2],
        [9, 0, 3, 3],
        [9, 1, 3, 3],
        [9, 2, 3, 3],
        [9, 3, 3, 3],
        [9, 5, 3, 3],
        [9, 0, 5, 5],
        [9, 1, 5, 5],
        [9, 2, 5, 2],
        [9, 3, 5, 3],
        [9, 5, 5, 5],
    ]
    for t in table_class:
        final[(class_nii == t[0]) & (class_sii == t[1]) & (class_oi == t[2])] = t[3]
    return final


# TEMPLATE for classifications
def diag_class_general(class_nii, class_sii, class_oi):
    """
    Final classification
    Codes:
    0 - Unclassified
    1 - SFN
    2 - Seyfert
    3 - LINER
    4 - TO
    5 - NLAGN (Seyfert or LINER)
    8 - TO or SFN
    9 - TO or NLAGN
    10 - Seyfert 1 (not used here)
    """
    final = np.zeros_like(class_sii)
    # Common classification for SII and OI
    # Key for sii oi class: [class sii, class oi, class sii oi]
    table_class = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 2, 1, 0],
        [0, 3, 1, 0],
        [0, 5, 1, 0],
        [0, 0, 2, 0],
        [0, 1, 2, 0],
        [0, 2, 2, 0],
        [0, 3, 2, 0],
        [0, 5, 2, 0],
        [0, 0, 3, 0],
        [0, 1, 3, 0],
        [0, 2, 3, 0],
        [0, 3, 3, 0],
        [0, 5, 3, 0],
        [0, 0, 5, 0],
        [0, 1, 5, 0],
        [0, 2, 5, 0],
        [0, 3, 5, 0],
        [0, 5, 5, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 2, 0, 0],
        [1, 3, 0, 0],
        [1, 5, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 2, 1, 0],
        [1, 3, 1, 0],
        [1, 5, 1, 0],
        [1, 0, 2, 0],
        [1, 1, 2, 0],
        [1, 2, 2, 0],
        [1, 3, 2, 0],
        [1, 5, 2, 0],
        [1, 0, 3, 0],
        [1, 1, 3, 0],
        [1, 2, 3, 0],
        [1, 3, 3, 0],
        [1, 5, 3, 0],
        [1, 0, 5, 0],
        [1, 1, 5, 0],
        [1, 2, 5, 0],
        [1, 3, 5, 0],
        [1, 5, 5, 0],
        [4, 0, 0, 0],
        [4, 1, 0, 0],
        [4, 2, 0, 0],
        [4, 3, 0, 0],
        [4, 5, 0, 0],
        [4, 0, 1, 0],
        [4, 1, 1, 0],
        [4, 2, 1, 0],
        [4, 3, 1, 0],
        [4, 5, 1, 0],
        [4, 0, 2, 0],
        [4, 1, 2, 0],
        [4, 2, 2, 0],
        [4, 3, 2, 0],
        [4, 5, 2, 0],
        [4, 0, 3, 0],
        [4, 1, 3, 0],
        [4, 2, 3, 0],
        [4, 3, 3, 0],
        [4, 5, 3, 0],
        [4, 0, 5, 0],
        [4, 1, 5, 0],
        [4, 2, 5, 0],
        [4, 3, 5, 0],
        [4, 5, 5, 0],
        [5, 0, 0, 0],
        [5, 1, 0, 0],
        [5, 2, 0, 0],
        [5, 3, 0, 0],
        [5, 5, 0, 0],
        [5, 0, 1, 0],
        [5, 1, 1, 0],
        [5, 2, 1, 0],
        [5, 3, 1, 0],
        [5, 5, 1, 0],
        [5, 0, 2, 0],
        [5, 1, 2, 0],
        [5, 2, 2, 0],
        [5, 3, 2, 0],
        [5, 5, 2, 0],
        [5, 0, 3, 0],
        [5, 1, 3, 0],
        [5, 2, 3, 0],
        [5, 3, 3, 0],
        [5, 5, 3, 0],
        [5, 0, 5, 0],
        [5, 1, 5, 0],
        [5, 2, 5, 0],
        [5, 3, 5, 0],
        [5, 5, 5, 0],
        [8, 0, 0, 0],
        [8, 1, 0, 0],
        [8, 2, 0, 0],
        [8, 3, 0, 0],
        [8, 5, 0, 0],
        [8, 0, 1, 0],
        [8, 1, 1, 0],
        [8, 2, 1, 0],
        [8, 3, 1, 0],
        [8, 5, 1, 0],
        [8, 0, 2, 0],
        [8, 1, 2, 0],
        [8, 2, 2, 0],
        [8, 3, 2, 0],
        [8, 5, 2, 0],
        [8, 0, 3, 0],
        [8, 1, 3, 0],
        [8, 2, 3, 0],
        [8, 3, 3, 0],
        [8, 5, 3, 0],
        [8, 0, 5, 0],
        [8, 1, 5, 0],
        [8, 2, 5, 0],
        [8, 3, 5, 0],
        [8, 5, 5, 0],
        [9, 0, 0, 0],
        [9, 1, 0, 0],
        [9, 2, 0, 0],
        [9, 3, 0, 0],
        [9, 5, 0, 0],
        [9, 0, 1, 0],
        [9, 1, 1, 0],
        [9, 2, 1, 0],
        [9, 3, 1, 0],
        [9, 5, 1, 0],
        [9, 0, 2, 0],
        [9, 1, 2, 0],
        [9, 2, 2, 0],
        [9, 3, 2, 0],
        [9, 5, 2, 0],
        [9, 0, 3, 0],
        [9, 1, 3, 0],
        [9, 2, 3, 0],
        [9, 3, 3, 0],
        [9, 5, 3, 0],
        [9, 0, 5, 0],
        [9, 1, 5, 0],
        [9, 2, 5, 0],
        [9, 3, 5, 0],
        [9, 5, 5, 0],
    ]

    for t in table_class:
        final[(class_nii == t[0]) & (class_sii == t[1]) & (class_oi == t[2])] = t[3]
    return final


# ----------------------------#
# Cid-Fernandes et al. 2011  #
# ----------------------------#


def diag_CidFernandes2011(x, c_x, ew_ha, c_ew_ha, ew_nii, c_ew_nii, use_limits=False):
    """
    Apply the diagnostic criterion of Cid-Fernandes et al. 2011
    Codes:
    0 - Unclassified
    1 - SFN
    2 - sAGN (Seyfert)
    3 - wAGN (LINER)
    4 - TO (not used here)
    5 - AGN (Seyfert or LINER)
    6 - RG Retired galaxy
    7 - PG Passive galaxy
    8 - TO or SFN (not used here)
    9 - TO or NLAGN (not used here)
    10 - Seyfert 1 (not used here)
    """
    conditions = [
        [((ew_ha < 0.5) | (ew_nii < 0.5)), 7],  # Passive galaxy
        [((ew_ha >= 0.5) & (ew_nii >= 0.5) & (ew_ha < 3)), 6],  # RG
        [((ew_ha >= 3) & (ew_nii >= 0.5) & (x <= -0.4)), 1],  # SFN
        [((ew_ha >= 3) & (ew_nii >= 0.5) & (x > -0.4) & (ew_ha < 6)), 3],  # wAGN
        [((ew_ha >= 6) & (ew_nii >= 0.5) & (x > -0.4)), 2],
    ]  # sAGN
    if not use_limits:
        cond_init = (
            (c_x == 0) & (c_ew_ha == 0) & (c_ew_nii == 0)
        )  # Only detections in the ratio and all the good EW
    else:
        cond_init = (
            (c_x >= 0) & (c_ew_ha >= 0) & (c_ew_nii >= 0)
        )  # Only well defined lines
        conditions_lim = [
            [
                (
                    ((c_x == 1) & (c_ew_ha == 0))
                    | ((c_x == 1) & (c_ew_ha == 2))
                    | ((c_x == 0) & (c_ew_ha == 2))
                )
                & ((ew_ha >= 3) & (ew_nii >= 0.5) & (x <= -0.4)),
                1,
            ],  # SFN (x<;y+ or x<;y^ or x+;y^)
            [
                (
                    ((c_x == 2) & (c_ew_ha == 0))
                    | ((c_x == 2) & (c_ew_ha == 2))
                    | ((c_x == 0) & (c_ew_ha == 2))
                )
                & ((ew_ha >= 6) & (ew_nii >= 0.5) & (x > -0.4)),
                2,
            ],  # sAGN (x>;y+ or x>;y^ or x+;y^)
            [
                (((c_x == 2) & (c_ew_ha == 0)))
                & ((ew_ha >= 3) & (ew_nii >= 0.5) & (x > -0.4) & (ew_ha < 6)),
                3,
            ],  # wAGN (x>;y+)
            [
                (((c_x == 2) & (c_ew_ha == 2)) | ((c_x == 0) & (c_ew_ha == 2)))
                & ((ew_ha >= 3) & (ew_nii >= 0.5) & (x > -0.4) & (ew_ha < 6)),
                5,
            ],  # AGN (x>;y^ or x+;y^)
            [
                (((c_x == 0) & (c_ew_ha == 1)) | ((c_x == 0) & (c_ew_ha == 1)))
                & ((ew_ha >= 3) & (ew_nii >= 0.5) & (x > -0.4) & (ew_ha < 6)),
                0,
            ],  # wAGN or passive NOT USED
            [
                (((c_x == 2) & (c_ew_ha == 0)))
                & ((ew_ha >= 0.5) & (ew_nii >= 0.5) & (ew_ha < 3)),
                6,
            ],  # RG (x>;y+)
            [
                (((c_x == 1) & (c_ew_ha == 0)) | ((c_x == 1) & (c_ew_ha == 1)))
                & ((ew_ha < 0.5) | (ew_nii < 0.5)),
                7,
            ],  # PG (x<;y+ or x<;yv)
            [
                (((c_x == 0) & (c_ew_ha == 1)) | ((c_x == 2) & (c_ew_ha == 1)))
                & (
                    ((ew_ha < 0.5) | (ew_nii < 0.5))
                    | ((ew_ha >= 0.5) & (ew_nii >= 0.5) & (ew_ha < 3))
                ),
                0,
            ],
        ]  # RG or PG NOT USED
        conditions.append(conditions_lim)
    return apply_conditions(cond_init, conditions)


#######################
# Auxiliary functions #
#######################


def numpy_arrays4(x, y, xx, yy):
    """
    Transform 4 values, tuples or lists into 4 numpy arrays.
    Useful for entering values for testing.
    """
    x = np.array(x)
    y = np.array(y)
    xx = np.array(xx)
    yy = np.array(yy)
    return x, y, xx, yy


def apply_conditions(cond_init, conditions):
    """
    Auxiliary function to apply the conditions.
    Returns diag and c_diag:
      * diag - diagnostic code
      * c_diag - code indicating if the diagnostic was applied to an element 1 or 0.
    """
    diag = np.zeros(len(cond_init), dtype="i")
    c_diag = np.zeros(len(cond_init), dtype="i")
    for cond, typec in conditions:
        diag[cond_init & cond] = typec
        c_diag[cond_init & cond] = 1
    return diag, c_diag
