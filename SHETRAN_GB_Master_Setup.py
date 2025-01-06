"""
-------------------------------------------------------------
SHETRAN GB Master Simulation Creator
-------------------------------------------------------------
Ben Smith, adapted from previous codes.
27/07/2022
-------------------------------------------------------------
This script should be a default script for generating SHETRAN files, it should be usable by everyone
for all setups. If you change the setup then please change this code so that is more generic (unless
that makes it super complex). This script often uses 'Try', which means that sometimes errors are hidden.
Look to change that over time.

--- USER INSTRUCTIONS:
The user should only have to edit:
 --- Set File Paths
 --- Set Processing Methods
All other code should be fixed. Make sure that you update the correct parts of the dictionaries.
'process_single_catchment["single"]' controls 'multiprocessing["process_multiple_catchments"]'

--- NOTES:
This code did not run well on the blades last time I tried it  (for the UDM setup) and created easting
and northing files. I wonder whether this is due to the wrong version of xarray, but that is just a guess.
This script uses...

--- MASKS:
Masks should be .asc
Mask paths used in multiprocessing will be:
  mask_folder_prefix + sim_mask_path + sim_name + Mask.txt
  Leave sim_mask_path in the csv blank if there are no additional bits to add in here.
  mask_folder_prefix can be changed to "" and the prefixed folder specified in the CSV
   instead if more specific naming convention is needed.
Masks MUST align with the nearest 1000m! Else your climate data will be blank.
Masks MUST NOT have isolated cells / diagonally connected cells. This will stop the
SHETRAN Prepare.exe script from setting up. Remove these manually (and check any
corresponding cell maps). Later version of SHETRAN Prepare may correct for this issue.

--- Resolution
The models can be set up in three resolutions: 1000m, 500m and 100m.

This is controlled simply by the mask, which must be of the desired resolution and align with
the 1000m, 500m or 100m grid. The input data should also match that resolution, this is done
automatically by selecting the appropriate data folder.

If the mask resolution does not match the static data resolution you will get an ERROR similar to:
    >> boolean index did not match indexed array along dimension 0;
    >> dimension is 12 but corresponding boolean dimension is 6
This is probably because one of the two have incorrect file paths.

--- Northern Ireland:
NI datasets are not fully included into the setup and so are processed differently. They
do not have gridded climate data, instead, this is available in processed form for a
selection of catchments (this was provided by Helen He at UEA). These only run until the
end of 2010. The Northern Ireland runs should run from 01/01/1980 - 01/01/2011. This was
incorrect in the initial version and the library files were corrected manually.
TODO:
 - Update the climate input data to most current period.
 - Update the scripts so that they can take the UKCP18 data.
 - Update the scripts to include other functions (e.g. UKCP18).
 - Test the multiprocessing - this wasn't used in anger with
   the historical runs, so check against UKCP18. Check q works.
 - Add a mask checker to ensure that the cells start on the 1000m
 - Consider that creating climate simulations with this will create
   incorrect model durations, as SHETRAN runs on calendar years, but
   climate years are only 360 days.
 - Load climate data in before running through catchments. As in 1a
   Quickload of the UKCP setup. One experiment with this managed to
   load all data and then cut it out, however the volume of data that
   was read in prior to the cutting was waaaaaay too large, causing
   python to crash.
 - Change the input DEM to have decimal places.
 - Consider changing Vegetation_Details.csv to LandCover_details.csv
 - RECENT:
     - Fix issues with the Xarray.
     - Check that the vegetation details are outputted.
     - Setup 100m.
-------------------------------------------------------------
"""

# --- Load in Packages ----------------------------------------
import SHETRAN_GB_Master_Setup_Functions as SF
import pandas as pd  # for reading CSVs
import numpy as np
import time
# import os
# import itertools

# -------------------------------------------------------------
# --- USER INPUTS ---------------------------------------------
# -------------------------------------------------------------

# --- Set File Paths -------------------------------------------

# Climate input folders:
create_climate_data = True
rainfall_input_folder = 'I:/CEH-GEAR downloads/'
temperature_input_folder = 'I:/CHESS_T/'
PET_input_folder = 'I:/CHESS/'

# Set Model period: 'yyyy-mm-dd'
start_time = '1980-01-01'
end_time = '2011-01-01'

# # Model Resolution:
# resolution = 1000  # [Cell size in meters - options are: 1000, 500, 100]
# resolution = resolution_string(resolution)

# Static Input Data Folder:
raw_input_folder = "I:/SHETRAN_GB_2021/02_Input_Data/Raw ASCII inputs for SHETRAN UK/500m/"


# --- Set Processing Methods -----------------------------------
# PYRAMID = 'C:/Users/nbs65/Newcastle University/PYRAMID - General/WP3/02 SHETRAN Simulations/'
process_single_catchment = dict(
    single=True,
    simulation_name='7006',
    mask_path="I:/SHETRAN_GB_2021/02_Input_Data/1kmBngMasks_Processed/7006_Mask.txt",
    # mask_path="S:/00 - Catchment Setups/Testing setup/7006.txt",
    output_folder="S:/00 - Catchment Setups/Testing setup_500m/")

# Choose Single / Multiprocessing:
multiprocessing = dict(
    process_multiple_catchments=not process_single_catchment["single"],
    simulation_list_csv='I:/SHETRAN_GB_2021/01_Scripts/Other/OFFLINE Generic Catchment Setup Script/Simulation_Setup_List.csv',
    mask_folder_prefix='I:/SHETRAN_GB_2021/02_Input_Data/1kmBngMasks_Processed/',  # 1kmBngMasks_Processed/', # I:\SHETRAN_GB_2021\02_Input_Data\superseded\1kmBngMasks
    output_folder_prefix="I:/SHETRAN_GB_2021/02_Input_Data/Catchment Maps - UDM/UDM_GB_LandCover_SSP4_2080/",  # 'I:/SHETRAN_GB_2021/02_Input_Data/NFM Catchment Maps/NFM_Maximum/',
    use_multiprocessing=True,  # May only work on Blades?
    n_processes=30,  # For use on the blades
    use_groups=True,  # [True, False][1]
    group="1")  # String. Not used when use_groups == False.


# --- Set Non-Default Land Cover Types -------------------------------------

# Urban development: if you wish to use land cover from the Newcastle University Urban Development Model (SELECT ONE).
Use_UrbanDevelopmentModel_2017 = False  # True / False (Default)
Use_UrbanDevelopmentModel_SSP2_2050 = False
Use_UrbanDevelopmentModel_SSP2_2080 = False
Use_UrbanDevelopmentModel_SSP4_2050 = False
Use_UrbanDevelopmentModel_SSP4_2080 = False

# Natural Flood Management: if you wish to use additional forest and storage from Sayers and Partners (SELECT ONE).
Use_NFM_Max_Woodland_Storage_Addition = False  # True /False (Default)
Use_NFM_Bal_Woodland_Storage_Addition = False


# -------------------------------------------------------------
# --- CALL FUNCTIONS FOR SETUP --------------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    if process_single_catchment["single"] == multiprocessing["process_multiple_catchments"]:
        print("WARNING - SINGLE AND MULTIPLE PROCESSES SELECTED. Check process_single_catchment & multiprocessing.")
        pass

    # --- Import the Static Dataset -------------------------------
    static_data = SF.read_static_asc_csv(static_input_folder=raw_input_folder,
                                         UDM_2017=Use_UrbanDevelopmentModel_2017,
                                         UDM_SSP2_2050=Use_UrbanDevelopmentModel_SSP2_2050,
                                         UDM_SSP2_2080=Use_UrbanDevelopmentModel_SSP2_2080,
                                         UDM_SSP4_2050=Use_UrbanDevelopmentModel_SSP4_2050,
                                         UDM_SSP4_2080=Use_UrbanDevelopmentModel_SSP4_2080,
                                         NFM_max=Use_NFM_Max_Woodland_Storage_Addition,
                                         NFM_bal=Use_NFM_Bal_Woodland_Storage_Addition)

    # --- Import the Climate Datasets -----------------------------
    # if create_climate_data:
    #
    #     # Find Climate Files to load for Each Variable:
    #     start_year = int(start_time[0:4])
    #     end_year = int(end_time[0:4])
    #
    #     print("  Reading rainfall...")
    #     prcp_input_files = SF.find_rainfall_files(start_year, end_year)
    #     rainfall_dataset = SF.read_climate_data(root_folder=rainfall_input_folder, filenames=prcp_input_files)
    #
    #     print("  Reading temperature...")
    #     tas_input_files = SF.find_temperature_or_PET_files(temperature_input_folder, start_year, end_year)
    #     temperature_dataset = SF.read_climate_data(root_folder=temperature_input_folder, filenames=tas_input_files)
    #
    #     print("  Reading PET...")
    #     pet_input_files = SF.find_temperature_or_PET_files(PET_input_folder, start_year, end_year)
    #     pet_dataset = SF.read_climate_data(root_folder=PET_input_folder, filenames=pet_input_files)
    #
    # else:
    #     rainfall_dataset = temperature_dataset = pet_dataset = None

    # --- Call Functions to Process a Single Catchment ------------
    if process_single_catchment["single"]:
        print("Processing single catchment...")
        SF.process_catchment(
            catch=process_single_catchment["simulation_name"],
            mask_path=process_single_catchment["mask_path"],
            simulation_startime=start_time, simulation_endtime=end_time,
            output_subfolder=process_single_catchment["output_folder"], static_inputs=static_data,
            produce_climate=create_climate_data, prcp_data_folder=rainfall_input_folder,
            tas_data_folder=temperature_input_folder, pet_data_folder=PET_input_folder)

    # --- Call Functions if Setting Up Multiple Catchments --------

    if multiprocessing["process_multiple_catchments"]:
        print("Processing multiple catchments...")

        # Read a list of simulation names to process:
        catchments_csv = pd.read_csv(multiprocessing["simulation_list_csv"], keep_default_na=False)

        # Check whether the simulation is in the group we're processing:
        if multiprocessing["use_groups"]:
            catchments_csv["Group"] = catchments_csv["Group"].astype('str')
            catchments_csv = catchments_csv[catchments_csv["Group"] == multiprocessing["group"]]

        # Get a list of the simulation/catchment names:
        simulation_names = list([str(c) for c in catchments_csv["Simulation_Name"]])

        # Create a list of file paths to the catchment masks:
        simulation_masks = [multiprocessing["mask_folder_prefix"] +
                            str(catchments_csv["Additional_Mask_Path"][x]) +
                            str(catchments_csv["Simulation_Name"][x]) + "_Mask.txt"
                            for x in catchments_csv["Simulation_Name"].index]

        # Create a list of output paths for the processed catchments:
        output_folders = [multiprocessing["output_folder_prefix"] +
                          str(catchments_csv["Additional_Output_Path"][x]) +
                          str(catchments_csv["Simulation_Name"][x]) + "/"
                          for x in catchments_csv.index]

        if multiprocessing["use_multiprocessing"]:
            print("Using multi-processing...")

            # Run the multiprocessing catchment setup:
            SF.process_mp(mp_catchments=simulation_names, 
                          mp_mask_folders=simulation_masks,
                          mp_output_folders=output_folders, 
                          mp_simulation_startime=start_time,
                          mp_simulation_endtime=end_time, 
                          mp_static_inputs=static_data,
                          mp_produce_climate=create_climate_data,
                          mp_prcp_data_folder=rainfall_input_folder,
                          mp_tas_data_folder=temperature_input_folder,
                          mp_pet_data_folder=PET_input_folder,
                          num_processes=multiprocessing["n_processes"])

        else:
            print("Using single processor...")
            for c in np.arange(0, len(simulation_names)):
                # Run the single processor catchment setup (for multiple catchments):
                SF.process_catchment(catch=simulation_names[c],
                                     mask_path=simulation_masks[c],
                                     simulation_startime=start_time,
                                     simulation_endtime=end_time,
                                     output_subfolder=output_folders[c],
                                     static_inputs=static_data,
                                     produce_climate=create_climate_data,
                                     prcp_data_folder=rainfall_input_folder,
                                     tas_data_folder=temperature_input_folder,
                                     pet_data_folder=PET_input_folder)
                time.sleep(1)
    print("Finished Processing Catchments")
