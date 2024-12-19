import os
import zipfile

import rasterio
from rasterio.merge import merge
import numpy as np

from scipy.ndimage import generic_filter

import geopandas as gpd

import pandas as pd

import rasterio.features
from shapely.geometry import box

root = 'I:/SHETRAN_GB_2021/02_Input_Data/National Data Inputs for SHETRAN UK/'
resolution_output = 500

def write_ascii(
        array: np,
        ascii_ouput_path: str,
        xllcorner: float,
        yllcorner: float,
        cellsize: float,
        ncols: int = None,
        nrows: int = None,
        NODATA_value: int = -9999,
        data_format: str = '%1.1f'):

        if len(array.shape) > 0:
            nrows, ncols = array.shape

        file_head = "\n".join(
            ["ncols         " + str(ncols),
             "nrows         " + str(nrows),
             "xllcorner     " + str(xllcorner),
             "yllcorner     " + str(yllcorner),
             "cellsize      " + str(cellsize),
             "NODATA_value  " + str(NODATA_value)])

        with open(ascii_ouput_path, 'wb') as output_filepath:
            np.savetxt(fname=output_filepath, X=array,
                       delimiter=' ', newline='\n', fmt=data_format, comments="",
                       header=file_head
                       )


def read_ascii_raster(file_path, data_type=int, return_metadata=True, replace_NA=False):
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
    if replace_NA:
       arr[arr==nodata] = np.nan

    headers = '\n'.join(headers)
    headers = headers.rstrip()

    if return_metadata:
        return arr, ncols, nrows, xll, yll, cellsize, nodata, headers, dc
    else:
        return arr

# Function for cell aggregation
def cell_reduce(array, block_size, func=np.nanmean):
    """
    Resample a NumPy array by reducing its resolution using block aggregation.
    Parameters:
    - array: Input NumPy array.
    - block_size: Factor by which to reduce the resolution.
    - func: Aggregation function (e.g., np.nanmean, np.nanmin, np.nanmax).
            Recomended to use nanmean etc. else you will lose coverage
    """
    shape = (array.shape[0] // block_size, block_size, array.shape[1] // block_size, block_size,)

    return func(array.reshape(shape), axis=(1, 3), )

# Define a function to calculate the mean of valid neighbors:
def fill_holes(values):
    # This will fill all holes with a value in a neighboring cell.

    center = values[4]  # Center pixel in the 3x3 window
    if np.isnan(center):  # If the center is a hole
        neighbors = values[np.arange(len(values)) != 4]  # Exclude the center
        valid_neighbors = neighbors[~np.isnan(neighbors)]  # Keep valid neighbors
        if len(valid_neighbors) > 0:  # Fill only if there are valid neighbors
            return valid_neighbors.mean()
    return center  # Return the original value if not a hole


# ----------------

# # Define the reclassification dictionary
# reclass_dict = {  # (CEH LCM to SHETRAN Classes)
#     1: 4, 2: 5, 3: 1,
#     4: 3, 5: 3, 6: 3, 7: 3,8: 3,
#     9: 6, 10: 6, 11: 6, 12: 6, 13: 6,
#     14: 2, 15: 2, 16: 2,  17: 2,  18: 2,  19: 2, 20: 2, 21: 2,
#     22: 7, 23: 7
# }

# List the shapefiles in GB:
# GB_LCM  = os.path.join(root, 'Land Use Inputs/LCM_2007_vector_GB_Digimap/lcm-2007-vec_5779248')
# GB_LCM_files = os.listdir(GB_LCM)
# shapefiles = [os.path.join(GB_LCM, sf) for sf in GB_LCM_files if sf.endswith('.shp')]

# NI_LCM = os.path.join(root, 'Land Use Inputs/LCM_2007_vector_NI_Digimap/lcm-2007-vec-ni_4578539')
# NI_LCM_files = os.listdir(NI_LCM)
# shapefiles = [os.path.join(NI_LCM, sf) for sf in NI_LCM_files if sf.endswith('.shp')]

# ----------------
# # Run through the files (including NI):
# counter = 1
# for shapefile in shapefiles:
#     print(counter, '/', len(shapefiles))
#     # Read in the data:
#     sf = gpd.read_file(shapefile)
#
#     # Reproject the Northern Ireland file into BNG (from ING):
#     if 'LCM_2007_vector_NI_Digimap' in shapefile:
#         sf = sf.to_crs(epsg=27700)
#
#     # Reclassify from LCM to SHETRAN classes'
#     sf['SHETRAN'] = sf['INTCODE'].map(reclass_dict)
#
#     # Cull the columns you don't need:
#     columns = sf.columns
#     columns = [column for column in columns if column not in ['SHETRAN', 'geometry']]
#     sf.drop(columns, inplace=True, axis=1)
#
#     # Dissolve the polygons to reduce file size:
#     sf_dissolved = sf.dissolve('SHETRAN')
#
#     # Save the updated shapefile:
#     sf_dissolved.to_file(
#         os.path.join(root, "Land Use Inputs/Reclassified shapefiles", os.path.basename(shapefile))
#     )
#
#     counter += 1

# ----------------

# # List the shapefiles in GB:
shapefile_path = os.path.join(root, 'Land Use Inputs/Reclassified shapefiles')
# shapefiles = os.listdir(shapefile_path)
# shapefiles = [os.path.join(shapefile_path, sf) for sf in shapefiles if sf.endswith('.shp')]
#
# # Merge into a single file:
# gdfs = []
# for shapefile in shapefiles:
#     print(shapefile)
#     sf = gpd.read_file(shapefile)
#     sf = sf.to_crs(epsg=27700)
#
#     gdfs.append(gpd.read_file(shapefile))
#
# # Merge all GeoDataFrames into one
# merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
#
# # Save the merged GeoDataFrame to a new shapefile
# merged_gdf.to_file(shapefile_path + '/Land_Use_GN_UK.shp')

# ----------------

# # Define parameters for the raster
# extent = (0, 0, 661000, 1241000)  # (minX, minY, maxX, maxY) in British National Grid (EPSG:27700)
# crs = "EPSG:27700"  # British National Grid CRS
#
# # Load the vector data (merged shapefile)
# shapefile = gpd.read_file(shapefile_path + '/Land_Use_GN_UK.shp')
#
# # Create raster dimensions based on the extent and resolution
# width = int((extent[2] - extent[0]) / resolution_output)
# height = int((extent[3] - extent[1]) / resolution_output)
#
# # Create an empty raster
# transform = rasterio.transform.from_bounds(*extent, width, height)
# raster_data = np.zeros((height, width), dtype=rasterio.uint8)
#
# # Rasterize the vector data
# rasterized = rasterio.features.rasterize(
#     ((geom, value) for geom, value in zip(shapefile.geometry, shapefile['SHETRAN'])),
#     out_shape=raster_data.shape,
#     transform=transform,
#     fill=-9999,  # Background value
#     dtype=rasterio.uint8
# )
#
# # Save the rasterized data to a GeoTIFF file
# with rasterio.open(
#     f'{shapefile_path}/Land Use {resolution_output}m.asc',
#     "w",
#     driver="AAIGrid",
#     height=height,
#     width=width,
#     count=1,
#     dtype=rasterized.dtype,
#     crs=crs,
#     transform=transform,
# ) as dst:
#     dst.write(rasterized, 1)

# ----------------

# extent = (0, 0, 661000, 1241000)  # (minX, minY, maxX, maxY) in British National Grid (EPSG:27700)
# crs = "EPSG:27700"  # British National Grid CRS

# Load the vector data (merged shapefile)
gdf = gpd.read_file(shapefile_path + '/LCM_2007_vector_UK_BNG.shp')

# Step 2: Create a vector grid
xmin, ymin, xmax, ymax = 0, 0, 661000, 1241000  # British National Grid boundaries
cell_size = resolution_output  # 100m resolution
cols = np.arange(xmin, xmax, cell_size)
rows = np.arange(ymin, ymax, cell_size)

grid_cells = []
for x in cols:
    for y in rows:
        grid_cells.append(box(x, y, x + cell_size, y + cell_size))

# Turn this into a geodataframe and give it an ID
grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=gdf.crs)
grid['ID'] = np.arange(0, grid.shape[0])

# Step 1: Intersect the grid and the shapefile
intersected = gpd.overlay(grid, gdf, how='intersection', keep_geom_type=False)

# Step 2: Calculate the area of each intersected polygon
intersected["area"] = intersected.area

# Step 3: Sort the intersected DataFrame by 'ID' and 'area' and crop to only the largest land type per cell:
intersected_sorted = intersected.sort_values(by=["ID", "area"], ascending=[True, False])

# Step 4: Drop duplicates based on 'ID', keeping only the largest area
filtered_intersected = intersected_sorted.drop_duplicates(subset="ID")
# filtered_intersected.to_file(shapefile_path + '/filtered_intersected.shp')

# 5. Converting filtered_intersected straight to raster misses cells, instead join the LC classes back to the grid:
# Perform the left join on the 'ID' column
grid_with_intersected = grid.merge(filtered_intersected[['SHETRAN', 'ID']], on="ID", how="left", suffixes=('_grid', '_intersected'))
# grid_with_intersected.to_file(shapefile_path + '/grid_with_intersected.shp')

# Step 6: Rasterize the intersected polygons
# Define the raster properties
transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, len(cols), len(rows))

# Prepare shapes and values for rasterisation:
shapes = ((geom, value) for geom, value in zip(grid_with_intersected.geometry, grid_with_intersected['SHETRAN']))

# Rasterize:
raster = rasterio.features.rasterize(
    shapes,
    out_shape=(len(rows), len(cols)),
    transform=transform,
    fill=-9999,  # NoData value
    dtype="int32"
)

# Convert 0s to -9999s for no data values:
raster[raster == 0] = -9999

write_ascii(
    array=raster,
    ascii_ouput_path=f'{root}/Processed Data/CEH_LCM_2007 {resolution_output}m.asc',
    xllcorner=xmin,
    yllcorner=ymin,
    cellsize=cell_size,
    NODATA_value=-9999,
    data_format='%1.0f'
)

# ----------------

