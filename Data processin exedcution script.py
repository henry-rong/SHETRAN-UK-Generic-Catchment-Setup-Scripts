import numpy as np
import rasterio.features
import math
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds

root = 'I:/SHETRAN_GB_2021/02_Input_Data/National Data Inputs for SHETRAN UK/'
resolution_output = 1000
print(resolution_output)


def write_ascii(array: np, ascii_ouput_path: str, xllcorner: float, yllcorner: float, cellsize: float,
                ncols: int = None, nrows: int = None, NODATA_value: int = -9999, data_format: str = '%1.1f'):
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
        arr[arr == nodata] = np.nan

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


# Define a function for getting the extents of a shapefile and expanding these to match a desired raster resolution:
def get_generous_bounds(shapefile, resolution):
    extents = shapefile.bounds

    xmin = math.floor(extents['minx'].min() / resolution) * resolution
    ymin = math.floor(extents['miny'].min() / resolution) * resolution
    xmax = math.ceil(extents['maxx'].max() / resolution) * resolution
    ymax = math.ceil(extents['maxy'].max() / resolution) * resolution

    return xmin, ymin, xmax, ymax


# Apply majority filter
def majority_filter(values):
    unique, counts = np.unique(values[values != -9999], return_counts=True)
    return unique[np.argmax(counts)] if len(unique) > 0 else -9999


# ----------------

# Reclassification mapping
reclass_mapping = {0: -9999,
                   1: 4, 2: 5, 3: 1, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 6,
                   10: 6, 11: 6, 12: 6, 13: 6, 14: 2, 15: 8, 16: 8,
                   17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 7, 23: 7
                   }

# Paths for your rasters
raster_GB_LCM = root + "/Land Use Inputs/LCM 2007 25m Raster/data/lcm2007gb25m.tif"
raster_NI_LCM = root + "/Land Use Inputs/LCM 2007 25m Raster/data/LCM2007_NI_25M_BNG.tif"

# Open LCM GB and NI raster files:
print('READING')
rasters = [rasterio.open(f) for f in [raster_GB_LCM, raster_NI_LCM]]

# Merge the rasters into a single UK raster:
merged_raster, merged_transform = merge(rasters)

# Create an empty array to hold the reclassified data:
reclassified_data = np.empty(merged_raster.shape)  # np.copy(merged_raster)

# Reclassify from the LCM classes the SHETRAN classes:
for original_value, new_value in reclass_mapping.items():
    reclassified_data[merged_raster == original_value] = new_value

# Change -9999s into nan values so that they do not influence processing:
reclassified_data[reclassified_data == -9999] = np.nan

# Set up an empty array to hole the resampled data:
xmin, ymin, xmax, ymax = 0, 0, 661000, 1241000  # resolution of the existing SHETRAN inputs
new_transform = from_bounds(xmin, ymin, xmax, ymax,
                            width=(xmax - xmin) // resolution_output,
                            height=(ymax - ymin) // resolution_output)
new_shape = ((ymax - ymin) // resolution_output, (xmax - xmin) // resolution_output)
resampled_raster = np.empty(new_shape)

# Resample the data to the desired resolution using the most common land use in each cell (the modal class):
reproject(  # You could also do this by applying the row_difference and cell_reduce method from the DEM.
    source=reclassified_data, destination=resampled_raster, src_transform=merged_transform,
    src_crs="EPSG:27700", dst_transform=new_transform, dst_crs="EPSG:27700",
    resampling=Resampling.mode  # Use the mode to get the value that is most common
)

# Change np.nan's back into -9999s:
resampled_raster[np.isnan(resampled_raster)] = -9999

# Write as an asc file:
output_path = f'{root}/Processed Data/UK Land Use {resolution_output}m'

# Save to file
with rasterio.open(
        output_path + '.asc', "w", driver="AAIGrid", height=resampled_raster.shape[0], width=resampled_raster.shape[1],
        count=1, dtype=resampled_raster.dtype, crs="EPSG:27700", transform=new_transform, nodata=-9999) as dst:
    dst.write(resampled_raster, 1)

# ----------------
