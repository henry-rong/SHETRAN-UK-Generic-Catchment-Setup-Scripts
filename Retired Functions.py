
def create_climate_files(climate_startime, climate_endtime, mask_path, catch, climate_output_folder,
                         prcp_folder, tas_folder, pet_folder):
    """
    Create climate time series.
    """

    start_year, _, _ = get_date_components(climate_startime)
    end_year, _, _ = get_date_components(climate_endtime)

    # Read catchment mask
    mask, ncols, nrows, xll, yll, cellsize, _, hdrs, _ = read_ascii_raster(
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
    prcp_input_files = find_rainfall_files(start_year, end_year)
    series_output_path = climate_output_folder + catch + '_Precip.csv'
    map_output_path = climate_output_folder + catch + '_Cells.asc'

    if not os.path.exists(series_output_path):
        make_series(
            ms_root_folder=prcp_folder, ms_filenames=prcp_input_files,
            xll=xll, yll=yll, urx=urx, ury=ury,
            ms_variable='rainfall_amount',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords, cell_ids=cell_ids, series_output_path=series_output_path,
            write_cell_id_map=True, map_output_path=map_output_path, map_hdrs=hdrs
        )

    # --- Temperature

    # Cell centre ll coords
    xll_centroid = xll + cellsize/2
    yll_centroid = yll + cellsize/2

    # Figure out coordinates of upper right
    urx_centroid = xll_centroid + (ncols - 1) * cellsize
    ury_centroid = yll_centroid + (nrows - 1) * cellsize

    # Get coordinates and IDs of cells inside catchment
    cat_coords_centroid = []
    for yv, xv in cat_coords:
        cat_coords_centroid.append((yv + cellsize/2, xv + cellsize/2))

    # Make temperature time series
    print("-------- Processing temperature data.")
    tas_input_files = find_CHESS_temperature_or_PET_files(tas_folder, start_year, end_year)
    series_output_path = climate_output_folder + catch + '_Temp.csv'
    if not os.path.exists(series_output_path):
        make_series(
            ms_root_folder=tas_folder, ms_filenames=tas_input_files,
            xll=xll_centroid, yll=yll_centroid, urx=urx_centroid, ury=ury_centroid,
            ms_variable='tas',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords_centroid, cell_ids=cell_ids, series_output_path=series_output_path
        )

    # --- PET
    print("-------- Processing evapotranspiration data.")

    # Make PET time series
    pet_input_files = find_CHESS_temperature_or_PET_files(pet_folder, start_year, end_year)
    series_output_path = climate_output_folder + catch + '_PET.csv'
    if not os.path.exists(series_output_path):
        make_series(
            ms_root_folder=pet_folder, ms_filenames=pet_input_files,
            xll=xll_centroid, yll=yll_centroid, urx=urx_centroid, ury=ury_centroid,
            ms_variable='pet',
            series_startime=climate_startime, series_endtime=climate_endtime,
            cat_coords=cat_coords_centroid, cell_ids=cell_ids, series_output_path=series_output_path
        )



def make_series(
        ms_root_folder, ms_filenames,
        xll, yll, urx, ury,
        ms_variable, series_startime, series_endtime,
        cat_coords, cell_ids, series_output_path, write_cell_id_map=False,
        map_output_path=None, map_hdrs=None):
    """
    Make and save climate time series for an individual variable.
    """

    if write_cell_id_map:
        np.savetxt(map_output_path, cell_ids, fmt='%d', header=map_hdrs, comments='')

    met_dataset = read_climate_data(root_folder=ms_root_folder, filenames=ms_filenames, variable=ms_variable,
                                    xll=xll, yll=yll, urx=urx, ury=ury)

    # sometimes pet is called peti, check that here just in case.
    if ms_variable == 'pet':
        ds_sel_var = list(met_dataset.keys())
        ms_variable = [ds_sel_var[i] for i in np.arange(0, len(ds_sel_var)) if 'pet' in ds_sel_var[i]][0]

    df = met_dataset[ms_variable].to_dataframe()
    df = df.unstack(level=['y', 'x'])

    y_coords = list(df.columns.levels[1])
    y_coords.sort(reverse=True)  # TODO: Check that this does actually want reversing.
    x_coords = list(df.columns.levels[2])
    x_coords.sort(reverse=False)

    dfs = df.loc[:, list(itertools.product([ms_variable], y_coords, x_coords))]

    # Subset on time period and cells in catchment
    # TODO Check that this doesn't delete data when cut out.
    # print("-------- Cropping", variable, "data to period.")
    df = dfs.sort_index().loc[series_startime:series_endtime]
    tmp = np.asarray(df.columns[:])
    # Get a list of all of the coordinates in the climate data:
    all_coords = [(y, x) for _, y, x in tmp]
    cat_indices = []
    ind = 0

    print(f'Testing: all_Coords = {all_coords}. Length = {len(all_coords)}')
    print('   ')
    print(f'Testing: cat_coords = {cat_coords}. Length = {len(cat_coords)}')

    # This loop will take each set of coordinates within the climate grid and check to see whether they are in the
    # list of cell coordinates within the catchment (cat_coords). If it is, then the cell number will be written to
    # the list cat_indices. This will result in a list if cell indexes for the climate data that are used to
    # subset the climate data and add it to the catchment csvs.

    # Run through the climate data, cell by cell:
    for all_pair in all_coords:
        # 1km SHETRAN - these should match nicely.
        if all_pair in cat_coords:
            cat_indices.append(ind)
            print('1000')
        # 500m SHETRAN
        if tuple([list(all_pair)[0]+500, list(all_pair)[1]]) in cat_coords:
            cat_indices.append(ind)
            print('500')
        if tuple([list(all_pair)[0], list(all_pair)[1]+500]) in cat_coords:
            cat_indices.append(ind)
            print('500')
        if tuple([list(all_pair)[0]+500, list(all_pair)[1]+500]) in cat_coords:
            cat_indices.append(ind)
            print('500')

        # And with smaller offset as temperature and pet coords are on offset grid.
        if tuple([list(all_pair)[0]-250, list(all_pair)[1]]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0], list(all_pair)[1]-250]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0]-250, list(all_pair)[1]-250]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0]+250, list(all_pair)[1]]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0], list(all_pair)[1]+250]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0]+250, list(all_pair)[1]+250]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0]-250, list(all_pair)[1]+250]) in cat_coords:
            cat_indices.append(ind)
            print('250')
        if tuple([list(all_pair)[0]+250, list(all_pair)[1]-250]) in cat_coords:
            cat_indices.append(ind)
            print('250')

        ind += 1

    df = df.iloc[:, cat_indices]

    print(f'\n Testing: cat_indices = {cat_indices}. Length = {len(cat_indices)}')

    # Convert from degK to degC if temperature
    if ms_variable == 'tas':
        df -= 273.15

    # Write outputs
    # print("-------- writing ", variable, "...")
    headers = np.unique(cell_ids)
    headers = headers[headers >= 1]
    print(f'\n Testing: headers (length {len(headers)}) = {headers}')
    print(f'\n Testing: headers assigned)')

    np.savetxt(map_output_path, cell_ids, fmt='%d', header=map_hdrs, comments='')

    df.to_csv(series_output_path, index=False, float_format='%.2f', header=headers)
    print(f'\n Testing: CSV written')


def read_climate_data(root_folder, filenames, variable, xll, yll, urx, ury,):
    first_loop = True

    # Run through the different decades, bolting the required catchment data into a common dataframe.
    for file in filenames:

        print("    - ", file)

        with xr.open_dataset(os.path.join(root_folder, file)) as ds:

            # Slice the dataset to only read in the necessary area of data:
            if variable == 'rainfall_amount':
                DS = ds.sel(y=slice(ury, yll),
                            x=slice(xll, urx))  # Y coords reversed as CHESS lists them backwards
            else:
                DS = ds.sel(y=slice(yll, ury), x=slice(xll, urx))

            if first_loop:
                DS_all_periods = DS
                first_loop = False
            else:
                DS_all_periods = xr.merge([DS_all_periods, DS])

    return DS_all_periods


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