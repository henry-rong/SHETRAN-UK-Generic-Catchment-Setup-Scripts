# -------------------------------------------------------------
# SHETRAN GB Master Simulation Creator
# -------------------------------------------------------------
# Ben Smith, adapted from previous codes.
# 27/07/2022
# -------------------------------------------------------------
#
# This script should be a default script for generating SHETRAN files, it should be usable by everyone
# for all setups. If you change the setup then please change this code so that is more generic (unless
# that makes it super complex). This script often uses 'Try', which means that sometimes errors are hidden.
# Look to change that over time.
#
# --- USER INSTRUCTIONS:
# The user should only have to edit:
#  --- Set File Paths
#  --- Set Processing Methods
# All other code should be fixed. Make sure that you update the correct parts of the dictionaries.
# 'process_single_catchment["single"]' controls 'multiprocessing["process_multiple_catchments"]'
#
# --- MASKS:
# Masks should be .asc
# Mask paths used in multiprocessing will be:
#   mask_folder_prefix + sim_mask_path + sim_name + Mask.asc.
#   Leave sim_mask_path in the csv blank if there are no additional bits to add in here.
#   mask_folder_prefix can be changed to "" and the prefixed folder specified in the CSV
#    instead if more specific naming convention is needed.
# Masks MUST align with the nearest 1000m! Else your climate data will be blank.
#
# TODO
# - Update the climate input data to most current period.
# - Update the scripts so that they can take the UKCP18 data.
# - Update the scripts to include other functions (e.g. UKCP18).
# - Test the multiprocessing - this wasn't used in anger with
#   the historical runs, so check against UKCP18. Check q works.
# - Test climate data == TRUE.
# - Add a mask checker to ensure that the cells start on the 1000m
# - Consider that creating climate simulations with this will create
#   incorrect model durations, as SHETRAN runs on calendar years, but 
#   Climate years are only 360 days.
#
# -------------------------------------------------------------


# --- Load in Packages ----------------------------------------
import numpy as np
import time
import SHETRAN_GB_Master_Setup_Functions as sf

# -------------------------------------------------------------
# --- USER INPUTS ---------------------------------------------
# -------------------------------------------------------------

# --- Set File Paths -------------------------------------------

# Climate input folders:
create_climate_data = True
rainfall_input_folder = 'I:/CEH-GEAR downloads/'
temperature_input_folder = 'I:/CHESS_T/'
PET_input_folder = 'I:/CHESS/'

# Model period: 'yyyy-mm-dd'
start_time = '2000-01-01'
end_time = '2015-12-30'

# Static Input Data Folder:
raw_input_folder = "I:/SHETRAN_GB_2021/inputs/Raw ASCII inputs for SHETRAN UK/"

# --- Set Processing Methods -----------------------------------
PYRAMID = 'C:/Users/nbs65/Newcastle University/PYRAMID - General/WP3/02 SHETRAN Simulations/'
process_single_catchment = dict(
    single=True,
    simulation_name='Tyne_at_Newcastle',
    mask_path=PYRAMID + "09_Tyne_at_Newcastle_03/Tyne_at_Newcastle_SetupMask.asc",
    output_folder=PYRAMID + "09_Tyne_at_Newcastle_05/")

# Choose Single / Multiprocessing:
multiprocessing = dict(
    process_multiple_catchments=not process_single_catchment["single"],
    simulation_list_csv='I:/SHETRAN_GB_2021/creation_test/Simulation_Setup_List.csv',
    mask_folder_prefix='I:/SHETRAN_GB_2021/creation_test/masks/',
    output_folder_prefix='I:/SHETRAN_GB_2021/creation_test/',
    use_multiprocessing=True, n_processes=2,
    use_groups=True, group=1)  # not used when use_groups == False.[True, False][1]

# -------------------------------------------------------------
# --- CALL FUNCTIONS FOR SETUP --------------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    if process_single_catchment["single"] == multiprocessing["process_multiple_catchments"]:
        print("WARNING - SINGLE AND MULTIPLE PROCESSES SELECTED. Check process_single_catchment & multiprocessing.")

    # --- Import the Static Dataset -------------------------------
    static_data = sf.read_static_asc_csv(raw_input_folder)

    # --- Call Functions to Process a Single Catchment ------------
    if process_single_catchment["single"]:
        print("Processing single catchment...")
        sf.process_catchment(
            catch=process_single_catchment["simulation_name"], mask_path=process_single_catchment["mask_path"],
            simulation_startime=start_time, simulation_endtime=end_time,
            output_subfolder=process_single_catchment["output_folder"], static_inputs=static_data,
            produce_climate=create_climate_data, prcp_input_folder=rainfall_input_folder,
            tas_input_folder=temperature_input_folder, pet_input_folder=PET_input_folder)

    # --- Multiprocess Catchments ---------------------------------

    if multiprocessing["process_multiple_catchments"]:
        print("Processing multiple catchments...")

        # Set up a list of catchments to process:
        simulation_names = []
        simulation_masks = []
        output_folders = []

        # Read a list of simulation names to process:
        with open(multiprocessing["simulation_list_csv"], 'r') as fh:
            # Read a line from the csv:
            fh.readline()
            for line in fh:
                sim_name, sim_mask_path, _, process_group, sim_out_path = line.rstrip().split(',')
                # Check whether the simulation is in the group we're processing:
                if multiprocessing["use_groups"]:
                    if int(process_group) != multiprocessing["group"]:
                        continue
                # If so then append it to the list:
                simulation_names.append(sim_name)
                simulation_masks.append(multiprocessing["mask_folder_prefix"] + sim_mask_path + sim_name + "_Mask.asc")
                output_folders.append(multiprocessing["output_folder_prefix"] + sim_out_path + sim_name + "/")

        if multiprocessing["use_multiprocessing"]:
            print("Using multi-processing...")

            sf.process_mp(mp_catchments=simulation_names, mp_mask_folders=simulation_masks,
                          mp_output_folders=output_folders, mp_simulation_startime=start_time,
                          mp_simulation_endtime=end_time, mp_static_inputs=static_data,
                          mp_produce_climate=create_climate_data, mp_prcp_input_folder=rainfall_input_folder,
                          mp_tas_input_folder=temperature_input_folder, mp_pet_input_folder=PET_input_folder,
                          num_processes=multiprocessing["n_processes"])

        else:
            print("Using single processing...")
            for c in np.arange(0, len(simulation_names)):
                sf.process_catchment(catch=simulation_names[c], mask_path=simulation_masks[c],
                                     simulation_startime=start_time, simulation_endtime=end_time,
                                     output_subfolder=output_folders[c], static_inputs=static_data,
                                     produce_climate=create_climate_data, prcp_input_folder=rainfall_input_folder,
                                     tas_input_folder=temperature_input_folder, pet_input_folder=PET_input_folder)
                time.sleep(1)
    print("Finished Processing Catchments")

