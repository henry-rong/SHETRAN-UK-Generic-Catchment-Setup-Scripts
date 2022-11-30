# -------------------------------------------------------------
# SHETRAN Generic Catchment Simulation Functions
# -------------------------------------------------------------
# Ben Smith, adapted from previous codes.
# 27/07/2022
# -------------------------------------------------------------
# This code holds the functions required for SHETRAN Generic
# Catchment Simulation Creator. Function updates should always
# be backwards compatible and as simple as possible to aid future
# users.
#
# Notes:
# CHESS rainfall reads in the Y coordinates backwards, if you
# change the meteorological inputs then check the coordinates.
# -------------------------------------------------------------

# --- Load in Packages ----------------------------------------
import os
import itertools
import xarray as xr
import pandas as pd
import copy
import datetime
import multiprocessing as mp
import numpy as np


# --- Create Functions ----------------------------------------
def read_ascii_raster(file_path, data_type=int, return_metadata=True):
    """
    Read ascii raster into numpy array, optionally returning headers.
    """
    headers = []
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            asc_line = fh.readline()
            headers.append(asc_line.rstrip())
            key, val = asc_line.rstrip().split()
            dc[key] = val
    ncols = int(dc['ncols'])
    nrows = int(dc['nrows'])
    xll = float(dc['xllcorner'])
    yll = float(dc['yllcorner'])
    cellsize = float(dc['cellsize'])
    nodata = float(dc['NODATA_value'])

    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)

    headers = '\n'.join(headers)
    headers = headers.rstrip()

    if return_metadata:
        return arr, ncols, nrows, xll, yll, cellsize, nodata, headers
    else:
        return arr


def get_catchment_coords_ids(xll, yll, urx, ury, cellsize, mask):
    """Find coordinates of cells in catchment and assign IDs."""
    x = np.arange(xll, urx + 1, cellsize)
    y = np.arange(yll, ury + 1, cellsize)
    y[::-1].sort()
    xx, yy = np.meshgrid(x, y)
    xx_cat = xx[mask == 0]
    yy_cat = yy[mask == 0]
    cat_coords = []
    for xv, yv in zip(xx_cat, yy_cat):
        cat_coords.append((yv, xv))

    cell_ids = np.zeros(xx.shape, dtype=int) - 9999
    counter = 1
    for i in range(len(y)):
        for j in range(len(x)):
            if mask[i, j] == 0:
                cell_ids[i, j] = counter
                counter += 1

    return cat_coords, cell_ids


def get_date_components(date_string, fmt='%Y-%m-%d'):
    date = datetime.datetime.strptime(date_string, fmt)
    return date.year, date.month, date.day


def get_veg_string(vegetation_array_for_library, static_input_dataset):
    """
    Get string containing vegetation details for the library file.
    """

    veg_vals = [int(v) for v in np.unique(vegetation_array_for_library[vegetation_array_for_library != -9999])]
    # strickler_dict = {1: 0.6, 2: 3, 3: 0.5, 4: 1, 5: 0.25, 6: 2, 7: 5}
    # strickler_dict

    # Extract the vegetation properties from the metadata
    veg_props = static_input_dataset.land_cover_lccs.attrs["land_cover_key"].loc[
        static_input_dataset.land_cover_lccs.attrs["land_cover_key"]["Veg Type #"].isin(veg_vals)].copy()
    # veg_props["strickler"] = [strickler_dict[item] for item in veg_props["Veg Type #"]]

    # Write the subset of properties out to a string
    veg_string = veg_props.to_csv(header=False, index=False)
    # veg_string = "<VegetationDetail>" + veg_string[:-1].replace("\n", "</VegetationDetail>\n<VegetationDetail>") +
    # "</VegetationDetail>\n"
    tmp = []
    for veg_line in veg_string[:-1].split('\n'):
        tmp.append('<VegetationDetail>' + veg_line.rstrip() + '</VegetationDetail>')
    veg_string = '\n'.join(tmp)
    return veg_string


def get_soil_strings(orig_soil_types, new_soil_types, static_input_dataset):
    """
    Get the unique soil columns out for the library file.
    """

    orig_soil_types = [int(v) for v in orig_soil_types]
    new_soil_types = [int(v) for v in new_soil_types]

    # Find the attributes of those columns
    soil_props = static_input_dataset.soil_type_APM.attrs["soil_key"].loc[
        static_input_dataset.soil_type_APM.attrs["soil_key"]["Soil Category"].isin(
            orig_soil_types)].copy()  # Change soil_type_APM to soil_type to use old soils!

    for orig_type, new_type in zip(orig_soil_types, new_soil_types):
        soil_props.loc[soil_props['Soil Category'] == orig_type, 'tmp0'] = new_type
    soil_props['Soil Category'] = soil_props['tmp0'].values
    soil_props['Soil Category'] = soil_props['Soil Category'].astype(int)

    # Rename the soil types for the new format of shetran
    soil_props["New_Soil_Type"] = soil_props["Soil Type"].copy()
    aquifer_types = ["NoGroundwater", "LowProductivityAquifer", "ModeratelyProductiveAquifer",
                     "HighlyProductiveAquifer"]

    soil_props['tmp1'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Top_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['tmp2'] = np.where(
        (~soil_props['Soil Type'].isin(aquifer_types)),
        'Sub_' + soil_props['Soil Type'],
        soil_props['Soil Type']
    )
    soil_props['New_Soil_Type'] = np.where(
        soil_props['Soil Layer'] == 1, soil_props['tmp1'], soil_props['tmp2']
    )

    # Assign a new soil code to the unique soil types
    soil_codes = soil_props.New_Soil_Type.unique()
    soil_codes_dict = dict(zip(soil_codes, [i + 1 for i in range(len(soil_codes))]))
    soil_props["Soil_Type_Code"] = [soil_codes_dict[item] for item in soil_props.New_Soil_Type]

    # Select the relevant information for the library file
    soil_types = soil_props.loc[:, ["Soil_Type_Code", "New_Soil_Type", "Saturated Water Content",
                                    "Residual Water Content", "Saturated Conductivity (m/day)",
                                    "vanGenuchten- alpha (cm-1)", "vanGenuchten-n"]]
    soil_types.drop_duplicates(inplace=True)

    soil_cols = soil_props.loc[:, ["Soil Category", "Soil Layer", "Soil_Type_Code", "Depth at base of layer (m)"]]

    # Write the subset of properties out to a string
    soil_types_string = soil_types.to_csv(header=False, index=False)
    soil_cols_string = soil_cols.to_csv(header=False, index=False)

    # soil_types_string = "<SoilProperty>" + soil_types_string[:-1].replace("\n", "</SoilProperty>\n<SoilProperty>")
    # + "</SoilProperty>\n" soil_cols_string = "<SoilDetail>" + soil_cols_string[:-1].replace("\n",
    # "</SoilDetail>\n<SoilDetail>") + "</SoilDetail>\n"

    tmp = []
    for line in soil_types_string[:-1].split('\n'):
        tmp.append('<SoilProperty>' + line.rstrip() + '</SoilProperty>')
    soil_types_string = '\n'.join(tmp)

    tmp = []
    for line in soil_cols_string[:-1].split('\n'):
        tmp.append('<SoilDetail>' + line.rstrip() + '</SoilDetail>')
    soil_cols_string = '\n'.join(tmp)

    return soil_types_string, soil_cols_string


def create_library_file(
        sim_output_folder, catch, veg_string, soil_types_string, soil_cols_string,
        sim_startime, sim_endtime, prcp_timestep=24, pet_timestep=24):
    """Create library file."""

    start_year, start_month, start_day = get_date_components(sim_startime)
    end_year, end_month, end_day = get_date_components(sim_endtime)

    output_list = [
        '<?xml version=1.0?><ShetranInput>',
        '<ProjectFile>{}_ProjectFile</ProjectFile>'.format(catch),
        '<CatchmentName>{}</CatchmentName>'.format(catch),
        '<DEMMeanFileName>{}_DEM.asc</DEMMeanFileName>'.format(catch),
        '<DEMminFileName>{}_MinDEM.asc</DEMMinFileName>'.format(catch),
        '<MaskFileName>{}_Mask.asc</MaskFileName>'.format(catch),
        '<VegMap>{}_LandCover.asc</VegMap>'.format(catch),
        '<SoilMap>{}_Soil.asc</SoilMap>'.format(catch),
        '<LakeMap>{}_Lake.asc</LakeMap>'.format(catch),
        '<PrecipMap>{}_Cells.asc</PrecipMap>'.format(catch),
        '<PeMap>{}_Cells.asc</PeMap>'.format(catch),
        '<VegetationDetails>',
        '<VegetationDetail>Veg Type #, Vegetation Type, Canopy storage capacity (mm), Leaf area index, '
        'Maximum rooting depth(m), AE/PE at field capacity,Strickler overland flow coefficient</VegetationDetail>',
        veg_string,
        '</VegetationDetails>',
        '<SoilProperties>',
        '<SoilProperty>Soil Number,Soil Type, Saturated Water Content, Residual Water Content, Saturated Conductivity '
        '(m/day), vanGenuchten- alpha (cm-1), vanGenuchten-n</SoilProperty> Avoid spaces in the Soil type names',
        soil_types_string,
        '</SoilProperties>',
        '<SoilDetails>',
        '<SoilDetail>Soil Category, Soil Layer, Soil Type, Depth at base of layer (m)</SoilDetail>',
        soil_cols_string,
        '</SoilDetails>',
        '<InitialConditions>0</InitialConditions>',
        '<PrecipitationTimeSeriesData>{}_Precip.csv</PrecipitationTimeSeriesData>'.format(catch),
        '<PrecipitationTimeStep>{}</PrecipitationTimeStep>'.format(prcp_timestep),
        '<EvaporationTimeSeriesData>{}_PET.csv</EvaporationTimeSeriesData>'.format(catch),
        '<EvaporationTimeStep>{}</EvaporationTimeStep>'.format(pet_timestep),
        '<MaxTempTimeSeriesData>{}_Temp.csv</MaxTempTimeSeriesData>'.format(catch),
        '<MinTempTimeSeriesData>{}_Temp.csv</MinTempTimeSeriesData>'.format(catch),
        '<StartDay>{}</StartDay>'.format(start_day, '02'),
        '<StartMonth>{}</StartMonth>'.format(start_month, '02'),
        '<StartYear>{}</StartYear>'.format(start_year),
        '<EndDay>{}</EndDay>'.format(end_day, '02'),
        '<EndMonth>{}</EndMonth>'.format(end_month, '02'),
        '<EndYear>{}</EndYear>'.format(end_year),
        '<RiverGridSquaresAccumulated>2</RiverGridSquaresAccumulated> Number of upstream grid squares needed to '
        'produce a river channel. A larger number will have fewer river channels.',
        '<DropFromGridToChannelDepth>2</DropFromGridToChannelDepth> The standard and minimum value is 2 if there are '
        'numerical problems with error 1060 this can be increased.',
        '<MinimumDropBetweenChannels>0.5</MinimumDropBetweenChannels> This depends on the grid size and how steep the '
        'catchment is. A value of 1 is a sensible starting point but more gently sloping catchments it can be reduced.',
        '<RegularTimestep>1.0</RegularTimestep> This is the standard Shetran timestep it is automatically reduced in '
        'rain. The standard value is 1 hour. The maximum allowed value is 2 hours.',
        '<IncreasingTimestep>0.05</IncreasingTimestep> speed of increase in timestep after rainfall back to the '
        'standard timestep. The standard value is 0.05. If if there are numerical problems with error 1060 it can be '
        'reduced to 0.01 but the simulation will take longer.',
        '<SimulatedDischargeTimestep>24.0</SimulatedDischargeTimestep> This should be the same as the measured '
        'discharge.',
        '<SnowmeltDegreeDayFactor>0.0002</SnowmeltDegreeDayFactor> Units  = mm s-1 C-1',
        '</ShetranInput>',
    ]
    output_string = '\n'.join(output_list)

    with open(sim_output_folder + catch + "_LibraryFile.xml", "w") as f:
        f.write(output_string)


def create_static_maps(static_input_dataset, xll, yll, ncols, nrows, cellsize,
                       static_output_folder, headers, catch, mask, nodata=-9999):
    """
    Write ascii files for DEM, minimum DEM, lake map, vegetation type map and soil map.
    """

    # Helper dictionary of details of static fields:
    # #- keys are names used for output and values are lists of variable name in
    # master static dataset alongside output number format
    static_field_details = {
        'DEM': ['surface_altitude', '%.2f'],
        'MinDEM': ['surface_altitude_min', '%.2f'],
        'Lake': ['lake_presence', '%d'],
        'LandCover': ['land_cover_lccs', '%d'],
        'Soil': ['soil_type_APM', '%d'],
    }

    xur = xll + (ncols * cellsize) - 1
    yur = yll + (nrows * cellsize) - 1

    catch_data = static_input_dataset.sel(y=slice(yur, yll), x=slice(xll, xur))

    # If we have additional data loaded in that we want to cut out, add these to the fields above:
    static_field_details_names = [v[0] for v in static_field_details.values()]
    static_input_data_names = list(catch_data.keys())
    names_to_add = [n for n in static_input_data_names if n not in static_field_details_names]
    for name in names_to_add:
        static_field_details[name] = [name, '%d']

    # Save each variable to ascii raster
    for array_name, array_details in static_field_details.items():
        array = copy.deepcopy(catch_data[array_details[0]].values)

        # Renumber soil types so consecutive from one
        if array_name == 'Soil':

            array_new = np.zeros(shape=array.shape)
            array_new[array_new == 0] = -9999

            orig_soil_types = np.unique(array[mask != nodata]).tolist()
            new_soil_types = range(1, len(orig_soil_types) + 1)
            for orig_type, new_type in zip(orig_soil_types, new_soil_types):
                array_new[array == orig_type] = new_type
            array = array_new

        array[mask == nodata] = nodata
        # if array_name in ['DEM', 'MinDEM']:
        #     array[mask <= 0] = 0.01

        # Remove any values <0 in the DEMs as these will crash the prepare.exe.
        if array_name in ['DEM', 'MinDEM']:
            array[(array > -9999) & (array <= 0)] = 0.01

        # Write the data out:
        map_output_path = static_output_folder + catch + '_' + array_name + '.asc'
        np.savetxt(
            map_output_path, array, fmt=array_details[1], header=headers, comments=''
        )

        # Get vegetation and soil type arrays for library file construction
        if array_name == 'LandCover':
            vegetation_arr = array
        if array_name == 'Soil':
            soil_arr = array

    # Also save mask
    np.savetxt(
        static_output_folder + catch + '_Mask.asc', mask, fmt='%d', header=headers,
        comments=''
    )

    return vegetation_arr, soil_arr, orig_soil_types, new_soil_types


def find_rainfall_files(year_from, year_to):
    x = [str(y) + '.nc' for y in range(year_from, year_to + 1)]
    return x


def find_temperature_or_PET_files(folder_path, year_from, year_to):
    files = sorted(os.listdir(folder_path))
    x = []
    for fn in files:
        if fn[-3:] == ".nc":
            if int(fn.split('_')[-1][:4]) in range(year_from, year_to + 1):
                x.append(fn)
    return x


def read_climate_data(root_folder, filenames):
    first_loop = True

    # Run through the different decades, bolting the required catchment data into a common dataframe.
    for file in filenames:

        print("    - ", file)

        with xr.open_dataset(os.path.join(root_folder, file)) as DS:

            if first_loop:
                DS_all_periods = DS
                first_loop = False
            else:
                DS_all_periods = xr.merge([DS_all_periods, DS])

    return DS_all_periods


def make_series(
        met_dataset,
        xll, yll, urx, ury,
        variable, series_startime, series_endtime,
        cat_coords, cell_ids, series_output_path, write_cell_id_map=False,
        map_output_path=None, map_hdrs=None):
    """
    Make and save climate time series for an individual variable.
    """

    # print("-------- Cropping", variable, "to catchment.")
    if variable == 'rainfall_amount':
        ds_sel = met_dataset.sel(y=slice(ury, yll),
                                 x=slice(xll, urx))  # Y coords reversed as CHESS lists them backwards
    else:
        ds_sel = met_dataset.sel(y=slice(yll, ury), x=slice(xll, urx))

    # sometimes pet is called peti, check that here just in case.
    if variable == 'pet':
        ds_sel_var = list(ds_sel.keys())
        variable = [ds_sel_var[i] for i in np.arange(0, len(ds_sel_var)) if 'pet' in ds_sel_var[i]][0]

    df = ds_sel[variable].to_dataframe()
    df = df.unstack(level=['y', 'x'])

    y_coords = list(df.columns.levels[1])
    y_coords.sort(reverse=True)  # TODO: Check that this does actually want reversing.
    x_coords = list(df.columns.levels[2])
    x_coords.sort(reverse=False)

    dfs = df.loc[:, list(itertools.product([variable], y_coords, x_coords))]

    # Subset on time period and cells in catchment
    # TODO Check that this doesn't delete data when cut out.
    # print("-------- Cropping", variable, "data to period.")
    df = dfs.sort_index().loc[series_startime:series_endtime]
    tmp = np.asarray(df.columns[:])
    all_coords = [(y, x) for _, y, x in tmp]
    cat_indices = []
    ind = 0
    for all_pair in all_coords:
        if all_pair in cat_coords:
            cat_indices.append(ind)
        ind += 1
    df = df.iloc[:, cat_indices]

    # Convert from degK to degC if temperature
    if variable == 'tas':
        df -= 273.15

    # Write outputs
    # print("-------- writing ", variable, "...")
    headers = np.unique(cell_ids)
    headers = headers[headers >= 1]

    df.to_csv(series_output_path, index=False, float_format='%.2f', header=headers)
    if write_cell_id_map:
        np.savetxt(map_output_path, cell_ids, fmt='%d', header=map_hdrs, comments='')


def create_climate_files(climate_startime, climate_endtime, mask_path, catch, climate_output_folder,
                         prcp_data, tas_data, pet_data):
    """
    Create climate time series.
    """

    start_year, _, _ = get_date_components(climate_startime)
    end_year, _, _ = get_date_components(climate_endtime)

    # Read catchment mask
    mask, ncols, nrows, xll, yll, cellsize, _, hdrs = read_ascii_raster(
        mask_path, data_type=int, return_metadata=True
    )

    # --- Precipitation

    # Figure out coordinates of upper right
    urx = xll + (ncols - 1) * cellsize
    ury = yll + (nrows - 1) * cellsize

    # Get coordinates and IDs of cells inside catchment
    cat_coords, cell_ids = get_catchment_coords_ids(xll, yll, urx, ury, cellsize, mask)

    # Make precipitation time series and cell ID map
    print("-------- Processing rainfall data.")
    series_output_path = climate_output_folder + catch + '_Precip.csv'
    map_output_path = climate_output_folder + catch + '_Cells.asc'

    if not os.path.exists(series_output_path):
        make_series(
            met_dataset=prcp_data,
            xll=xll, yll=yll, urx=urx, ury=ury,
            variable='rainfall_amount',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords, cell_ids=cell_ids, series_output_path=series_output_path,
            write_cell_id_map=True, map_output_path=map_output_path, map_hdrs=hdrs
        )

    # --- Temperature

    # Cell centre ll coords
    xll_centroid = xll + 500.0
    yll_centroid = yll + 500.0

    # Figure out coordinates of upper right
    urx_centroid = xll_centroid + (ncols - 1) * cellsize
    ury_centroid = yll_centroid + (nrows - 1) * cellsize

    # Get coordinates and IDs of cells inside catchment
    cat_coords_centroid = []
    for yv, xv in cat_coords:
        cat_coords_centroid.append((yv + 500.0, xv + 500.0))

    # Make temperature time series
    print("-------- Processing temperature data.")
    series_output_path = climate_output_folder + catch + '_Temp.csv'
    if not os.path.exists(series_output_path):
        make_series(
            met_dataset=tas_data,
            xll=xll_centroid, yll=yll_centroid, urx=urx_centroid, ury=ury_centroid,
            variable='tas',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords_centroid, cell_ids=cell_ids, series_output_path=series_output_path
        )

    # --- PET
    print("-------- Processing evapotranspiration data.")

    # Make PET time series
    series_output_path = climate_output_folder + catch + '_PET.csv'
    if not os.path.exists(series_output_path):
        make_series(
            met_dataset=pet_data,
            xll=xll_centroid, yll=yll_centroid, urx=urx_centroid, ury=ury_centroid,
            variable='pet',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords_centroid, cell_ids=cell_ids, series_output_path=series_output_path
        )


def process_catchment(
        catch, mask_path, simulation_startime, simulation_endtime, output_subfolder, static_inputs,
        produce_climate=True, prcp_data=None, tas_data=None, pet_data=None  # ,q=None
):
    """
    Create all files needed to run shetran-prepare.
    produce_climate is true or false option. If False, climate files will not be created.
    """

    if not os.path.isdir(output_subfolder):
        os.mkdir(output_subfolder)

    try:
        # Read mask
        print(catch, ": reading mask...")
        mask, ncols, nrows, xll, yll, cellsize, _, headers = read_ascii_raster(
            mask_path, data_type=int, return_metadata=True)

        # Create static maps and return vegetation_array (land cover) and soil arrays/info
        print(catch, ": creating static maps...")
        vegetation_array, soil_array, orig_soil_types, new_soil_types = create_static_maps(
            static_inputs, xll, yll, ncols, nrows, cellsize, output_subfolder, headers, catch, mask)

        # Create climate time series files (and cell ID map)
        if produce_climate:
            print(catch, ": Creating climate files...")
            create_climate_files(simulation_startime, simulation_endtime, mask_path, catch, output_subfolder,
                                 prcp_data, tas_data, pet_data)

        # Get strings of vegetation and soil properties/details for library file
        # print(catch, ": creating vegetation (land use) and soil strings...")
        veg_string = get_veg_string(vegetation_array, static_inputs)
        soil_types_string, soil_cols_string = get_soil_strings(orig_soil_types, new_soil_types, static_inputs)

        # Create library file
        # print(catch, ": creating library file...")
        create_library_file(output_subfolder, catch, veg_string, soil_types_string, soil_cols_string,
                            simulation_startime, simulation_endtime)

        # sys.exit()

    except Exception as E:
        print(E)
        pass


def process_mp(mp_catchments, mp_mask_folders, mp_output_folders, mp_simulation_startime,
               mp_simulation_endtime, mp_static_inputs, mp_prcp_data, mp_tas_data,
               mp_pet_data, mp_produce_climate=False, num_processes=10):
    manager = mp.Manager()
    # q = manager.Queue()
    pool = mp.Pool(num_processes)

    jobs = []
    for catch in np.arange(0, len(mp_catchments)):
        job = pool.apply_async(process_catchment,
                               (mp_catchments[catch], mp_mask_folders[catch], mp_simulation_startime,
                                mp_simulation_endtime, mp_output_folders[catch], mp_static_inputs, mp_produce_climate,
                                mp_prcp_data, mp_tas_data, mp_pet_data))

        jobs.append(job)

    for job in jobs:
        job.get()

    # q.put('kill')
    pool.close()
    pool.join()


def read_static_asc_csv(static_input_folder,
                        UDM_2017=False, UDM_2050=False, UDM_2080=False,
                        NFM_max=False, NFM_balanced=False):
    """
    This functions will load in the raw data for the UK, i.e. asc and csv files, and convert these to the dictionary
    object used in the setups. There should be 7 files with the following names, all in the same folder (argument):
        - SHETRAN_UK_DEM.asc
        - SHETRAN_UK_minDEM.asc
        - SHETRAN_UK_lake_presence.asc
        - SHETRAN_UK_LandCover.asc
        - Vegetation_Details.csc
        - SHETRAN_UK_SoilGrid_APM.asc
        - SHETRAN_UK_SoilDetails.csc

        - UDM_GB_LandCover_2017.asc
        - UDM_GB_LandCover_2050.asc
        - UDM_GB_LandCover_2080.asc

        - NFMmax_GB_Woodland.asc
        - NFMmax_GB_Storage.asc
        - NFMbalanced_GB_Woodland.asc
        - NFMmax_GB_Storage.asc

    All .asc files should have the same extents and cell sizes.

    :param static_input_folder:
    :param UDM_2017: True or False depending on whether you want to use the default CEH 2007 or the UDM baseline map.
    :param UDM_2050: True or False depending on whether you want to use the default CEH 2007 or the UDM 2050 map.
    :param UDM_2080: True or False depending on whether you want to use the default CEH 2007 or the UDM 2080 map.
    :param NFM_balanced:
    :param NFM_max:
    :return:
    """

    # Raise an error if there are multiple land covers selected:
    if (UDM_2017 + UDM_2050 + UDM_2080) > 1:
        raise ValueError("Multiple UDM land cover maps are 'True'; only a single map can be used.")

    # Load in the coordinate data (assumes all data has same coordinates:
    _, ncols, nrows, xll, yll, cellsize, _, _ = read_ascii_raster(static_input_folder + "SHETRAN_UK_DEM.asc",
                                                                  return_metadata=True)

    # Create eastings and northings. Note, the northings are reversed to match the maps
    eastings = np.arange(xll, ncols * cellsize + yll, cellsize)
    northings = np.arange(yll, nrows * cellsize + yll, cellsize)[::-1]
    eastings_array, northings_array = np.meshgrid(eastings, northings)

    # Set the desired land cover:
    if UDM_2017:
        LandCoverMap = "UDM_GB_LandCover_2017.asc"
    elif UDM_2050:
        LandCoverMap = "UDM_GB_LandCover_2050.asc"
    elif UDM_2080:
        LandCoverMap = "UDM_GB_LandCover_2080.asc"
    else:
        LandCoverMap = "SHETRAN_UK_LandCover.asc"

    # Create xarray database to load/store the static input data:
    ds = xr.Dataset({
        "surface_altitude": (["y", "x"],
                             np.loadtxt(static_input_folder + "SHETRAN_UK_DEM.asc", skiprows=6),
                             {"units": "m"}),
        "surface_altitude_min": (["y", "x", ],
                                 np.loadtxt(static_input_folder + "SHETRAN_UK_minDEM.asc", skiprows=6),
                                 {"units": "m"}),
        "lake_presence": (["y", "x"],
                          np.loadtxt(static_input_folder + "SHETRAN_UK_lake_presence.asc", skiprows=6)),
        "land_cover_lccs": (["y", "x"],
                            np.loadtxt(static_input_folder + LandCoverMap, skiprows=6),
                            {"land_cover_key": pd.read_csv(static_input_folder + "Vegetation_Details.csv")}),
        "soil_type_APM": (["y", "x"],
                          np.loadtxt(static_input_folder + "SHETRAN_UK_SoilGrid_APM.asc", skiprows=6),
                          {"soil_key": pd.read_csv(static_input_folder + "SHETRAN_UK_SoilDetails.csv")})},
        coords={"easting": (["y", "x"], eastings_array, {"projection": "BNG"}),
                "northing": (["y", "x"], northings_array, {"projection": "BNG"}),
                "x": (["x"], eastings, {"projection": "BNG"}),
                "y": (["y"], northings, {"projection": "BNG"})})

    # Load in the GB NFM Max map from Sayers and Partners:
    if NFM_max:
        ds["NFM_max_storage"] = (["y", "x"],
                                 np.loadtxt(static_input_folder + "NFMmax_GB_Storage.asc", skiprows=6))
        ds["NFM_max_woodland"] = (["y", "x"],
                                  np.loadtxt(static_input_folder + "NFMmax_GB_Woodland.asc", skiprows=6))
    if NFM_balanced:
        ds["NFM_balanced_storage"] = (["y", "x"],
                                      np.loadtxt(static_input_folder + "NFMbalanced_GB_Storage.asc", skiprows=6))
        ds["NFM_balanced_woodland"] = (["y", "x"],
                                       np.loadtxt(static_input_folder + "NFMbalanced_GB_Woodland.asc", skiprows=6))

    return ds
