# IMPORT AND CONVERT TO NUMPY ARRAY - FUNCTION FROM TASGEOEXAMPLE
import rasterio
import numpy as np
import os  
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import generic_filter
import cupy as cp
from scipy.stats import mode



raster_map = r"C:\Users\TyHow\Documents\3. Work\GIS Stuff\SCP testing\calingasta_best_output.tif"

# read geotiffs
data, names = [], []  # Lists to store data and corresponding file names

with rasterio.open(raster_map, 'r') as src:  # Open GeoTIFF file for reading
    # read spatial information
    transform = src.transform  # Get affine transformation matrix
    region = (src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3])  # Get bounding box coordinates (left, bottom, right, top)
    # read band 1 data
    d = src.read(1).astype('float32')  # Read data from the first band and convert to float32
    nodata_mask = d == src.nodata  # Create a mask for NoData values
    d[nodata_mask] = np.nan  # Replace NoData values with NaN
    # append data to lists
    data.append(d)  # Append data to the list
    names.append(os.path.basename(raster_map).replace('.tif',''))  # Append file name to the list (without extension)

# stack list into 3D numpy array
data = np.stack(data)  # Stack the list of arrays into a 3D numpy array
data.shape, names  # Return the shape of the data array and the list of file names


# Define the filter function that computes the mode
def filter_function(invalues):
    # Calculate the mode, ignoring NaNs
    invalues_mode = mode(invalues, axis=None, nan_policy='omit')
    return invalues_mode.mode[0]  # Return the mode value

# Apply the mode filter using generic_filter
smoothed_array = generic_filter(data, function=filter_function, size=3)


# Plot the filtered raster array
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('tab20', len(np.unique(smoothed_array[~np.isnan(smoothed_array)])))

plt.imshow(smoothed_array, cmap=cmap, interpolation='nearest')
plt.colorbar(label='Classes')
plt.title('Smoothed Classified Raster Plot')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
