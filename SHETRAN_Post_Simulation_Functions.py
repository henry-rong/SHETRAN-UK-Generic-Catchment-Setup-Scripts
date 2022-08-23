# -------------------------------------------------------------
# SHETRAN Post Simulation Functions
# -------------------------------------------------------------
# Ben Smith
# 08/08/2022
# -------------------------------------------------------------
# This code is written to provide users with a range of
# functions that are generally useful for manipulating and
# analysing SHETRAN simulations. These include data extraction,
# analysis and visualisation.
# -------------------------------------------------------------


# --- Load in Packages ----------------------------------------
import hydroeval as he  # https://pypi.org/project/hydroeval/ - open conda prompt: pip install hydroeval
import os
import pandas as pd
import numpy as np
import shutil
import datetime


# --- Calculate Objective Functions for Flows -----------------
def shetran_obj_functions(regular_simulation_discharge_path: str, recorded_discharge_path: str,
                          start_date: str, period: list = None, recorded_date_discharge_columns: list = None,
                          return_flows=False, return_period=False):
    """
    Notes:
    - Assumes daily flow data, can be altered within function.
    - Assumes that recorded flows have dates and are regularly spaced, with no gaps.
    - NAs will be skipped from the analysis. NA count will be returned.

    TODO - consider whether you can add code that allows you to take other columns
            from the record so that they can be visualised at the end.

    regular_simulation_discharge_path:  Path to the txt file
    recorded_discharge_path:            Path to the csv file
    start_date:                         The start date of the simulated flows: "DD-MM-YYYY"
    period:                             The period to use (i.e. calibration/validation) as a list of dates:
                                        ["YYY-MM-DD", "YYY-MM-DD"].
                                        Leave blank if you want to use the whole thing.
                                        Leave as single item in list if you want to use until the end of the data.
    recorded_date_discharge_columns:    The columns (as a list) that contain the date and then flow data.
    RETURNS:                            The NSE value as an array.
    """

    # --- Read in the flows for Sim and Rec:
    if recorded_date_discharge_columns is None:
        recorded_date_discharge_columns = ["date", "discharge_vol"]

    flow_rec = pd.read_csv(recorded_discharge_path,
                           usecols=recorded_date_discharge_columns,
                           parse_dates=[recorded_date_discharge_columns[0]])

    # Set the columns to the following so that they are always correctly referenced:
    # (Do not use recorded_date_discharge_columns!)
    flow_rec.columns = ["date", "discharge_vol"]
    flow_rec = flow_rec.set_index('date')

    # Read in the simulated flows:
    flow_sim = pd.read_csv(regular_simulation_discharge_path)
    flow_sim.columns = ["flow"]

    # --- Give the simulation dates:
    flow_sim['date'] = pd.date_range(start=start_date, periods=len(flow_sim), freq='D')
    flow_sim = flow_sim.set_index('date').shift(-1)
    # ^^ The -1 removes the 1st flow, which is the flow before the simulation.

    # --- Resize them to match
    flows = flow_sim.merge(flow_rec, on="date")
    # ^^ Merge removes the dates that don't coincide. Beware missing record data!

    # Select the period for analysis (if given):
    if period is not None:
        if len(period) == 1:
            flows = flows[flows.index >= period[0]]
        if len(period) == 2:
            flows = flows[(flows.index >= period[0]) & (flows.index <= period[1])]

    # --- Do the comparison
    flow_NAs = np.isnan(flows["discharge_vol"])  # The NAs are actually automatically removed

    # Calculate the objective function:
    obj_funs = {"NSE": np.round(he.evaluator(he.nse, flows["flow"], flows["discharge_vol"]), 2),
                "KGE": np.round(he.evaluator(he.kge, flows["flow"], flows["discharge_vol"]), 2),
                "RMSE": np.round(he.evaluator(he.rmse, flows["flow"], flows["discharge_vol"]), 2),
                "PBias": np.round(he.evaluator(he.pbias, flows["flow"], flows["discharge_vol"]), 2)}

    # Print out the % of data that are NA:
    print(str(round(len(np.arange(len(flow_NAs))[flow_NAs]) / len(flows) * 100, 3)) + "% of comparison data are NA")

    if (period is not None) & (return_period):
        obj_funs["period"] = period

    if return_flows:
        obj_funs["flows"] = flows

    return obj_funs


# --- Sweep Files from Blades to Folder -----------------------
def folder_copy(source_folder, destination_folder, overwrite=False, outputs_only=False, complete_only=False):
    """
    "I:/SHETRAN_GB_2021/scripts/Blade_Sweeper.py" will execute this function for the Blades and CONVEX.

    :param source_folder: E.g. "C:/BenSmith/Blade_SHETRANGB_OpenCLIM_UKCP18rcm_220708_APM/Temp_simulations/"
    :param destination_folder: E.g. "I:/SHETRAN_GB_2021/UKCP18rcm_220708_APM_GB/"
    :param overwrite: For if you want to overwrite the destination folder (False/True)
    :param outputs_only: For if you only want to copy "outputs_..." files (False/True)
    :param complete_only: For if you only want to copy completed files, based on PRI file (False/True)
    :return: A list of copied files
    """

    # Check whether the destination folder exists (make it if not):
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    # Get a list of the folders to copy:
    files_2_copy = os.listdir(source_folder)

    # Set conditions to skip incomplete simulations if desired:
    if complete_only:
        pri_file = [i for i in files_2_copy if i.endswith("pri.txt")]

        # If there isn't a PRI file, skip the copy:
        if len(pri_file) == 0:
            return source_folder + " was not copied as it is incomplete."

        # If there is, then check completeness:
        with open(source_folder + pri_file[0], 'r') as f:
            lines = f.read().split("\n")
            comp_line = lines[-24]
            if not comp_line.startswith("Normal completion of SHETRAN run:"):
                # If incomplete, skip the copy, else continue:
                return source_folder + " was not copied as it is incomplete."

    # If NOT overwriting files, remove duplicates from source list:
    if not overwrite:
        destination_files = os.listdir(destination_folder)
        files_2_copy = [i for i in files_2_copy if i not in destination_files]

    # If you only want to copy outputs, only include these in the copy list:
    if outputs_only:
        files_2_copy = [i for i in files_2_copy if "output" in i]

    # Copy each of the remaining files across:
    if len(files_2_copy) > 0:
        for file in files_2_copy:
            shutil.copy2(source_folder + file, destination_folder + file)
        return files_2_copy
    else:
        return "No files to copy..."


# --- Get Date Components from a data string ------------------
def get_date_components(date_string, fmt='%Y-%m-%d'):
    # "1980/01/01"
    date = datetime.datetime.strptime(date_string, fmt)
    return date.year, date.month, date.day



