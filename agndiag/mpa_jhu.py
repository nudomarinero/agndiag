"""
Auxiliary methods to clean MPA-JHU line data.
The lines have to be entered into a pandas dataframe.
"""
__author__ = "jsm"
import numpy as np
from .lineclass import (
    diag_nii_Sabater2012,
    diag_oi_Sabater2012,
    diag_sii_Sabater2012,
    diag_CidFernandes2011,
    diag_class_OiSiiNiiMine,
    diag_class_Sabater2012,
)


name_lines = [
    "H_BETA",
    "OIII_5007",
    "OI_6300",
    "H_ALPHA",
    "NII_6584",
    "SII_6717",
    "SII_6731",
]
name_params = ["_FLUX", "_FLUX_ERR", "_CONT", "_CONT_ERR"]
# name_out_params = ["flux","e_flux","c_flux","ew","e_ew","c_ew"]
name_ratios = ["oiii_h_beta", "nii_h_alpha", "oi_h_alpha", "sii_h_alpha"]
dict_ratios = {
    "oiii_h_beta": ["OIII_5007", "H_BETA"],
    "nii_h_alpha": ["NII_6584", "H_ALPHA"],
    "oi_h_alpha": ["OI_6300", "H_ALPHA"],
    "sii_h_alpha": ["SII", "H_ALPHA"],
}
# name_ratios_params = ["ratio","e_ratio","c_ratio"]
name_diagnostic = ["nii", "oi", "sii"]
dict_diagnostic = {
    "nii": ["nii_h_alpha", "oiii_h_beta"],
    "oi": ["oi_h_alpha", "oiii_h_beta"],
    "sii": ["sii_h_alpha", "oiii_h_beta"],
}
cor_factor = [1.882, 1.566, 1.378, 2.473, 2.039, 1.621, 1.621]


def clean_data(df, sigma=3.0, ew_method=1):
    """
    Clean the line data.
    Applies the correction factors to the errors.
    """
    global name_lines, cor_factor
    # TODO: Check that the lines are in the dataframe
    for i, line in enumerate(name_lines):
        ## Flux
        flux = df[line + "_FLUX"]
        e_flux = df[line + "_FLUX_ERR"] * cor_factor[i]
        c_flux = df[line + "_FLUX"].astype(int)
        c_flux[~np.isnan(c_flux)] = 0
        # Bad lines
        # TODO: Use nan
        cond = e_flux <= 0.0  # Bad lines
        flux[cond] = 0.0
        e_flux[cond] = 0.0
        c_flux[cond] = -1
        # Lines detected below 3 sigma
        cond = (np.abs(flux) < sigma * e_flux) & (c_flux != -1)
        flux[cond] = sigma * e_flux[cond]
        c_flux[cond] = 1
        # Add cleaned flux
        df["flux_" + line] = flux
        df["e_flux_" + line] = e_flux
        df["c_flux_" + line] = c_flux

        ## Equivalent width
        cont = df[line + "_CONT"]
        e_cont = df[line + "_CONT_ERR"] * cor_factor[i]
        ew = np.zeros_like(cont)
        e_ew = np.zeros_like(cont)
        c_ew = df[line + "_CONT"].astype(int)
        c_ew[~np.isnan(c_ew)] = 0
        # Bad lines in EW
        cond = (c_flux == -1) | (e_cont <= 0.0) | (cont <= 0.0)
        cont[cond] = 0.0
        e_ew[cond] = 0.0
        c_ew[cond] = -1
        # Only for good lines
        cond = c_ew != -1
        ew[cond] = flux[cond] / cont[cond]
        e_ew[cond] = np.sqrt(
            (e_flux[cond] / cont[cond]) ** 2
            + (flux[cond] * e_cont[cond] / (cont[cond] ** 2)) ** 2
        )
        cond = (flux < sigma * e_flux) & (cont >= sigma * e_cont) & (c_ew != -1)
        c_ew[cond] = 2
        cond = (flux >= sigma * e_flux) & (cont < sigma * e_cont) & (c_ew != -1)
        c_ew[cond] = 1
        cond = (flux < sigma * e_flux) & (cont < sigma * e_cont) & (c_ew != -1)
        c_ew[cond] = 3
        # Add cleaned EW
        df["ew_" + line] = ew
        df["e_ew_" + line] = e_ew
        df["c_ew_" + line] = c_ew


def combine_sii(df, param="flux"):
    """
    Combine the params (flux or ew) of the two SII lines
    """
    assert param in ["flux", "ew"]
    df[param + "_SII"] = df[param + "_SII_6717"] + df[param + "_SII_6717"]
    df["e_" + param + "_SII"] = (
        df["e_" + param + "_SII_6717"] + df["e_" + param + "_SII_6717"]
    )
    code1 = df["c_" + param + "_SII_6717"]
    code2 = df["c_" + param + "_SII_6717"]
    df["c_" + param + "_SII"] = code1
    df["c_" + param + "_SII"][(code1 > 0) | (code2 > 0)] = 1
    df["c_" + param + "_SII"][(code1 < 0) | (code2 < 0)] = -1


def get_ratios(df):
    """
    Get the line ratios used in the diagnostic diagrams.
    """
    global name_ratios, dict_ratios
    combine_sii(df)
    for i, ratio in enumerate(name_ratios):
        flux_num = df["flux_" + dict_ratios[ratio][0]]
        flux_den = df["flux_" + dict_ratios[ratio][1]]
        e_flux_num = df["e_flux_" + dict_ratios[ratio][0]]
        e_flux_den = df["e_flux_" + dict_ratios[ratio][1]]
        c_flux_num = df["c_flux_" + dict_ratios[ratio][0]]
        c_flux_den = df["c_flux_" + dict_ratios[ratio][1]]
        log_num_den = np.log10(flux_num / flux_den)
        e_log_num_den = (
            1.0
            / np.log(10.0)
            * np.sqrt((e_flux_num / flux_num) ** 2 + (e_flux_den / flux_den) ** 2)
        )
        c_log_num_den = df["c_flux_" + dict_ratios[ratio][0]].astype(int)
        c_log_num_den[~np.isnan(c_log_num_den)] = 0  # Default zero for detections
        c_log_num_den[(c_flux_num > 0) & (c_flux_den == 0)] = 1  # Upper limit
        c_log_num_den[(c_flux_num == 0) & (c_flux_den > 0)] = 2  # Lower limit
        c_log_num_den[(c_flux_num > 0) & (c_flux_den > 0)] = 3  # Non-defined
        c_log_num_den[(flux_num < 0.0) | (flux_den < 0.0)] = -2  # Negative flux
        c_log_num_den[(c_flux_num < 0) | (c_flux_den < 0)] = -1  # Flagged line
        # Add the ratios
        df[ratio] = log_num_den
        df["e_" + ratio] = e_log_num_den
        df["c_" + ratio] = c_log_num_den


def apply_diag_CidFernandes2011(df):
    """
    Obtain the classification from the Cid-Fernandes diagnostic diagrams.
    Appends to 'diagnostic' an array with the classification in each diagnostic diagram.
    Also appends to 'diagnostic' an array indicating if the galaxy was classified.
    """
    name = "CidFernandes2011"
    # if self.ratios == {}:
    #     self.get_ratios()
    x = df["nii_h_alpha"]
    c_x = df["c_nii_h_alpha"]
    ew_ha = df["ew_H_ALPHA"]
    c_ew_ha = df["c_ew_H_ALPHA"]
    ew_nii = df["ew_NII_6584"]
    c_ew_nii = df["c_ew_NII_6584"]
    diag, c_diag = diag_CidFernandes2011(x, c_x, ew_ha, c_ew_ha, ew_nii, c_ew_nii)
    df["whan_" + name] = diag
    df["c_whan_" + name] = c_diag


def get_diag_ratios(df, diag_name):
    global dict_diagnostic
    x = df[dict_diagnostic[diag_name][0]]
    y = df[dict_diagnostic[diag_name][1]]
    c_x = df["c_" + dict_diagnostic[diag_name][0]]
    c_y = df["c_" + dict_diagnostic[diag_name][1]]
    return x, y, c_x, c_y


def apply_diag_Sabater2012(df, use_limits=True):
    """
    Obtain the classification from the three diagnostic diagrams.
    Appends to 'diagnostic' three arrays with the classification in each diagnostic diagram.
    Also appends to 'diagnostic' three arrays indicating if the galaxy was classified.
    """
    name = "Sabater2012"
    # if self.ratios == {}:
    #     self.get_ratios()
    # NII diagnostic
    df["nii_" + name], df["c_nii_" + name] = diag_nii_Sabater2012(
        *get_diag_ratios(df, "nii"), use_limits=use_limits
    )
    # SII diagnostic
    df["sii_" + name], df["c_sii_" + name] = diag_sii_Sabater2012(
        *get_diag_ratios(df, "sii"), use_limits=use_limits
    )
    # OI diagnostic
    df["oi_" + name], df["c_oi_" + name] = diag_oi_Sabater2012(
        *get_diag_ratios(df, "oi"), use_limits=use_limits
    )
    # Final classification
    df["class_" + name], df["class_to_" + name] = diag_class_Sabater2012(
        df["nii_" + name], df["sii_" + name], df["oi_" + name]
    )
