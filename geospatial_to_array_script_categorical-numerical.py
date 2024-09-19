#IMPORTS

from typing import Tuple, Dict, Any, List
import os
from tkinter import Tk, filedialog, simpledialog, Toplevel, Button, Checkbutton, IntVar, Label, Frame
from tkinter.filedialog import askopenfilenames, askdirectory, asksaveasfilename

import cupy as cp
import geopandas as gpd
import hvplot.xarray
import ipywidgets as widgets
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
import panel as pn
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import stats
from scipy.ndimage import generic_filter
from scipy.stats import mode
import shapely.geometry as sg
from shapely.geometry import box
import xarray as xr

from IPython.display import display

from joblib import Parallel, delayed

import easygui

#%%

#CALCULATE GRID SIZE (from mask short edge)

# Function to compute grid size based on the mask file
def compute_grid_size(geojson_file: str, short_edge_cells: int = 20) -> Tuple[int, int]:
    # Read the GeoJSON file using GeoPandas
    gdf = gpd.read_file(geojson_file)
    
    # Get the bounding box of the masking region
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Calculate width and height of the bounding box
    width = maxx - minx
    height = maxy - miny

    # Determine which is the short and long edge
    if width < height:
        short_edge = width
        long_edge = height
        orientation = 'portrait'
    else:
        short_edge = height
        long_edge = width
        orientation = 'landscape'

    # Compute the aspect ratio
    aspect_ratio = long_edge / short_edge

    # Compute the number of cells for the long edge
    long_edge_cells = int(short_edge_cells * aspect_ratio)

    # Determine the grid size based on the orientation
    if orientation == 'portrait':
        grid_size = (short_edge_cells, long_edge_cells)
    else:
        grid_size = (long_edge_cells, short_edge_cells)

    return grid_size

# Prompt the user for the short_edge_cells value using tkinter
root = Tk()
root.withdraw()  # Hide the root window

# Ask the user for the short edge size
short_edge_cells = simpledialog.askinteger("Input", "Enter the number of cells for the short edge:", minvalue=1)

root.destroy()  # Close the tkinter root window

if short_edge_cells is None:
    raise ValueError("You must enter a valid number for the short edge size.")

# Set the mask file path
mask_file = r"C:\Users\TyHow\Documents\3. Work\GIS Stuff\ML_pilot_data\MASK.geojson"

# Compute grid size using the mask file
grid_size = compute_grid_size(mask_file, short_edge_cells=short_edge_cells)[::-1]
print(f"Calculated grid size: {grid_size}")

#%%

#SELECT VECTOR FILES

# Function to interactively select files
def select_geojson_files():
    # Create a Tkinter root window (hidden)
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring the dialog to the front

    # Open the file selection dialog
    geojson_files = filedialog.askopenfilenames(
        title="Select GeoJSON Files",
        filetypes=[("GeoJSON files", "*.geojson"), ("All files", "*.*")]
    )
    
    root.destroy()  # Close the root window after selection
    return list(geojson_files)  # Convert tuple to list and return

# Use the function to select files
geojson_files = select_geojson_files()

# Print the selected files for verification
print(f"Selected GeoJSON files: {geojson_files}")

#SELECT VECTOR FILE LAYERS
# Function to select features for each GeoJSON file
def select_columns(gdf_dict):
    """
    Create a tkinter GUI window to allow the user to select which columns (features) to process.
    """
    selected_features = {}  # To store selected features for each file
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Bring the dialog to the front

    # Function to create the selection window
    def create_selection_window():
        selection_window = Toplevel(root)
        selection_window.title("Select Features to Process")

        # Store variables for checkboxes
        feature_vars = {}

        # Create checkboxes for each file and its features
        for geojson_file, gdf in gdf_dict.items():
            feature_columns = [col for col in gdf.columns if col != gdf.geometry.name]

            # Extract just the file name from the full path
            file_name = os.path.basename(geojson_file)

            # Add a label for each file
            file_label = Label(selection_window, text=f"File: {file_name}")
            file_label.pack(anchor='w', padx=10, pady=5)

            # Create checkboxes for each feature in the file
            feature_vars[geojson_file] = {}
            for feature in feature_columns:
                var = IntVar()
                checkbutton = Checkbutton(selection_window, text=feature, variable=var)
                checkbutton.pack(anchor='w')
                feature_vars[geojson_file][feature] = var

        # Function to handle "OK" button click
        def on_ok():
            for geojson_file, features in feature_vars.items():
                selected_features[geojson_file] = [feature for feature, var in features.items() if var.get() == 1]
            selection_window.destroy()
            root.quit()  # Properly close the Tkinter main loop

        # Add an "OK" button
        Button(selection_window, text="OK", command=on_ok).pack(pady=10)

        selection_window.mainloop()

    create_selection_window()
    root.destroy()  # Ensure the root window is properly destroyed
    return selected_features

# Load the GeoJSON files into GeoDataFrames
gdf_dict = {file: gpd.read_file(file) for file in geojson_files}

# Use the function to allow the user to select columns for each file
vector_features_to_process = select_columns(gdf_dict)

# Print the selected features for verification
print(f"Selected features to process: {vector_features_to_process}")

#%%

#VECTOR PROCESSING FUNCTIONS

# Function to process each cell in the grid
def process_cell(idx, cell, gdf, sindex, feature_column, category_to_int, filename_prefix):
    i, j = divmod(idx, grid_size[1])

    # Use the spatial index to find potential intersecting polygons
    possible_matches_index = list(sindex.intersection(cell.bounds))
    if not possible_matches_index:
        return i, j, np.nan

    possible_matches = gdf.iloc[possible_matches_index]
    if possible_matches.empty:
        return i, j, np.nan

    intersections = possible_matches.geometry.intersection(cell)
    valid_intersections = intersections[intersections.area > 0]

    if valid_intersections.empty:
        return i, j, np.nan

    if len(valid_intersections) > 5:
        areas_per_category = {}
        for idx, intersection in enumerate(valid_intersections):
            if not intersection.is_empty:
                category = possible_matches.iloc[idx][feature_column]
                category_key = f"{filename_prefix}_{category}"
                if category_key not in areas_per_category:
                    areas_per_category[category_key] = 0
                areas_per_category[category_key] += intersection.area

        if areas_per_category:
            max_category = max(areas_per_category, key=areas_per_category.get)
            return i, j, category_to_int[max_category]
        else:
            return i, j, np.nan
    else:
        largest_intersection_idx = valid_intersections.area.idxmax()
        category = possible_matches.loc[largest_intersection_idx, feature_column]
        return i, j, category_to_int[f"{filename_prefix}_{category}"]

# Function to process each feature column
def process_feature_column(geojson_file, feature_column, grid_size, target_crs, filename_prefix, x, y):
    gdf = gpd.read_file(geojson_file)
    print(f"Processing feature column: {feature_column} from file: {geojson_file}")

    gdf = gdf.to_crs(target_crs)
    if gdf.empty:
        print(f"GeoDataFrame for {geojson_file} is empty after reprojecting. Skipping column: {feature_column}")
        return None

    # Check if the geometry is a point
    geom_type = gdf.geometry.geom_type.iloc[0]

    if geom_type == "Point" or geom_type == "MultiPoint":
        print("Detected point geometry, buffering...")
        gdf["geometry"] = gdf.geometry.buffer(1000)  # Apply a buffer of 1000 meters to points

    unique_categories = gdf[feature_column].unique()
    print(f"Unique categories in {feature_column}: {unique_categories}")
    category_to_int = {f"{filename_prefix}_{cat}": i for i, cat in enumerate(unique_categories)}

    grid = np.full(grid_size, np.nan)
    sindex = gdf.sindex

    cells = [box(x[j], y[i], x[j + 1], y[i + 1])
             for i in range(grid_size[0])
             for j in range(grid_size[1])]

    results = Parallel(n_jobs=-1)(delayed(process_cell)(
        idx, cell, gdf, sindex, feature_column, category_to_int, filename_prefix
    ) for idx, cell in enumerate(cells))

    if not results:
        print(f"No results were generated for feature column: {feature_column} from file: {geojson_file}")
        return None

    for i, j, value in results:
        grid[i, j] = value

    # Flip the grid vertically (along Y-axis)
    grid_flipped = np.flipud(grid)

    return (f"{filename_prefix}_{feature_column}", grid_flipped, category_to_int)

# Batch processing function
def geojson_to_numpy_grid_3d_batch(
    grid_size: Tuple[int, int],  # Grid size for the output array
    geojson_files: List[str],  # List of GeoJSON files
    features_to_process: Dict[str, List[str]],  # Dictionary with filenames as keys and list of features as values
    target_crs: str = "EPSG:3857"  # Web Mercator projection
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[Any, int]], List[Dict[str, Any]]]:
    all_feature_grids = {}
    all_feature_mappings = {}
    geospatial_info_list = []

    results = []

    # Process each file and its corresponding features
    for geojson_file in geojson_files:
        # Get the filename without extension for prefixing
        filename_prefix = os.path.splitext(os.path.basename(geojson_file))[0]

        # Read the GeoJSON file to get the total bounds
        gdf = gpd.read_file(geojson_file)
        gdf = gdf.to_crs(target_crs)
        minx, miny, maxx, maxy = gdf.total_bounds

        x = np.linspace(minx, maxx, grid_size[1] + 1)
        y = np.linspace(miny, maxy, grid_size[0] + 1)

        # Extract relevant features for this file
        file_features = features_to_process.get(geojson_file, [])

        # Store geospatial information for each file
        geospatial_info = {
            'transform': (minx, miny, maxx, maxy),
            'crs': target_crs,
            'file_name': filename_prefix
        }
        geospatial_info_list.append(geospatial_info)

        # Use joblib to parallelize the processing of each feature column
        results.extend(Parallel(n_jobs=-1)(delayed(process_feature_column)(
            geojson_file, feature_column, grid_size, target_crs, filename_prefix, x, y
        ) for feature_column in file_features))

    for feature_name, grid, category_to_int in results:
        all_feature_grids[feature_name] = grid
        all_feature_mappings[feature_name] = category_to_int

    grid_3d = np.stack(list(all_feature_grids.values()), axis=0)

    return grid_3d, all_feature_grids, all_feature_mappings, geospatial_info_list


#%%

#%%

# SELECT/PROCESS RASTERS

# CUDA Median Filter for Categorical Rasters
median_filter_kernel = cp.RawKernel(r'''
extern "C" __global__
void median_filter(const float* input, float* output, int width, int height, int window_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_window = window_size / 2;
    float window[1024];

    if (x < width && y < height) {
        int count = 0;
        for (int i = -half_window; i <= half_window; ++i) {
            for (int j = -half_window; j <= half_window; ++j) {
                int nx = x + j;
                int ny = y + i;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float val = input[ny * width + nx];
                    if (!isnan(val)) {
                        window[count] = val;
                        count++;
                    }
                }
            }
        }
        for (int i = 0; i < count - 1; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    float temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }
        if (count > 0) {
            output[y * width + x] = window[count / 2];
        } else {
            output[y * width + x] = __int_as_float(0x7fffffff);
        }
    }
}
''', 'median_filter')

# Helper Functions for Processing
def preprocess_data(input_array):
    input_array = np.array(input_array)
    input_array = np.where(input_array == "nan", np.nan, input_array)
    input_array = input_array.astype(np.float32)
    return input_array

# Processing Categorical Rasters
def process_categorical_rasters(categorical_raster_files, window_size, grid_size, raster_target_transform, raster_target_crs):
    categorical_data = []
    raster_feature_mappings = []
    
    for layer_index, raster_file in enumerate(categorical_raster_files):
        with rasterio.open(raster_file, 'r') as src:
            print(f"Processing categorical file: {raster_file}")
            categorical_array = np.full(grid_size, np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=categorical_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=raster_target_transform,
                dst_crs=raster_target_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
            preprocessed_data = preprocess_data(categorical_array)

            block_size = (16, 16)
            smoothed_categorical_data = np.empty_like(preprocessed_data)

            data_layer_gpu = cp.array(preprocessed_data)
            grid_size_gpu = (
                (data_layer_gpu.shape[1] + block_size[0] - 1) // block_size[0],
                (data_layer_gpu.shape[0] + block_size[1] - 1) // block_size[1]
            )
            output_gpu = cp.empty_like(data_layer_gpu)
            median_filter_kernel(grid_size_gpu, block_size, (data_layer_gpu, output_gpu, data_layer_gpu.shape[1], data_layer_gpu.shape[0], window_size))
            smoothed_layer = cp.asnumpy(output_gpu)
            smoothed_categorical_data = smoothed_layer

            # Append the processed data and mappings
            categorical_data.append(smoothed_categorical_data)
            file_name = os.path.basename(raster_file).replace('.tiff', '').replace('.tif', '')
            raster_feature_mappings.append((file_name, layer_index))
    
    categorical_data = np.stack(categorical_data, axis=0)
    print(f"Categorical raster stack shape: {categorical_data.shape}")
    return categorical_data, raster_feature_mappings

# Numerical Raster Processing
def process_numerical_rasters(numerical_raster_files, grid_size, raster_target_transform, raster_target_crs):
    numerical_data = []
    raster_feature_mappings = []
    
    for layer_index, raster_file in enumerate(numerical_raster_files):
        with rasterio.open(raster_file, 'r') as src:
            print(f"Processing numerical file: {raster_file}")
            numerical_array = np.full(grid_size, np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=numerical_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=raster_target_transform,
                dst_crs=raster_target_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
            numerical_data.append(numerical_array)

            file_name = os.path.basename(raster_file).replace('.tiff', '').replace('.tif', '')
            raster_feature_mappings.append((file_name, layer_index))

    numerical_data = np.stack(numerical_data, axis=0)
    print(f"Numerical raster stack shape: {numerical_data.shape}")
    return numerical_data, raster_feature_mappings

# User Input and File Selection
def select_rasters_and_window():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    numerical_raster_files = askopenfilenames(
        title="Select Numerical Raster Files",
        filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")]
    )

    categorical_raster_files = askopenfilenames(
        title="Select Categorical Raster Files",
        filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")]
    )

    window_size = simpledialog.askinteger("Input", "Enter the window size for the batch median filter:", initialvalue=20)

    root.destroy()
    return numerical_raster_files, categorical_raster_files, window_size


numerical_raster_files, categorical_raster_files, window_size = select_rasters_and_window()

raster_gdf = gpd.read_file(mask_file)
minx, miny, maxx, maxy = raster_gdf.total_bounds
raster_gdf = raster_gdf.to_crs("EPSG:4326")
raster_target_transform = from_bounds(minx, miny, maxx, maxy, grid_size[1], grid_size[0])
raster_target_crs = "EPSG:4326"

# Process numerical rasters
numerical_data, numerical_raster_mappings = process_numerical_rasters(numerical_raster_files, grid_size, raster_target_transform, raster_target_crs)

# Process categorical rasters
categorical_data, categorical_raster_mappings = process_categorical_rasters(categorical_raster_files, window_size, grid_size, raster_target_transform, raster_target_crs)

# Combine all layers (vector, numerical, and categorical)
raster_data = np.concatenate((numerical_data, categorical_data), axis=0)
raster_feature_mappings = numerical_raster_mappings + categorical_raster_mappings
raster_names = [os.path.basename(file).replace('.tif', '') for file in numerical_raster_files + categorical_raster_files]

print(f"Combined array shape: {raster_data.shape}")
print("Layer Name Mapping List:", raster_names)
print("Raster Feature Mappings:", raster_feature_mappings)




#%%

#PROCESS VECTORS

vector_data, vector_feature_grids, vector_feature_mappings, vector_geospatial_info_list = geojson_to_numpy_grid_3d_batch(grid_size, geojson_files, vector_features_to_process)

# Print results
print("Shape of the 3D grid array:", vector_data.shape)
print("Feature grids:", vector_feature_grids.keys())
print("Feature mappings:", vector_feature_mappings)
print("Geospatial information for each file:", vector_geospatial_info_list)


#%%

#SAVE RASTER DATA

# Hide the root window for file dialog
root = Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)  # Bring the file dialog to the front

# Prompt the user to select a folder to save the files
output_directory = askdirectory(
    initialdir=r"C:\Users\TyHow\Documents\3. Work\ML_test_area\exports",
    title="Select a Folder to Save Output Rasters (BEWARE OVERWRITE!)"
)

if output_directory:
    # Construct file paths using the selected folder and default file names
    output_6_rasters_file = os.path.join(output_directory, "output_rasters.npy")
    output_6_rasters_layer_mappings_file = os.path.join(output_directory, "output_rasters_layer_mappings.npy")

    # Save the files
    np.save(output_6_rasters_file, raster_data)
    np.save(output_6_rasters_layer_mappings_file, raster_feature_mappings)

    print(f"Files saved in: {output_directory}")

# Destroy the root window after file dialogs are closed
root.destroy()

#%%

#SAVE VECTOR DATA

# Hide the root window for file dialog
root = Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)  # Bring the file dialog to the front

# Prompt the user to select a folder to save the files
output_directory = askdirectory(
    initialdir=r"C:\Users\TyHow\Documents\3. Work\ML_test_area\exports",
    title="Select a Folder to Save Output Vectors (BEWARE OVERWRITE!)"
)

if output_directory:
    # Construct file paths using the selected folder and default file names
    output_array_file = os.path.join(output_directory, "output_vectors.npy")
    output_feature_grid_file = os.path.join(output_directory, "output_vector_feature_grid.npy")
    output_feature_mappings_file = os.path.join(output_directory, "output_vector_feature_mappings.npy")
    output_geospatial_info_file = os.path.join(output_directory, "output_vector_geospatial_info.npy")

    # Save the files
    np.save(output_array_file, vector_data)
    np.save(output_feature_grid_file, vector_feature_grids)
    np.save(output_feature_mappings_file, vector_feature_mappings)
    np.save(output_geospatial_info_file, vector_geospatial_info_list)

    print(f"Files saved in: {output_directory}")

# Destroy the root window after file dialogs are closed
root.destroy()

#%%


#COMBINE ARRAYS - WITH LAYER NAMES

# Combine the arrays (vector and raster)
if vector_data.shape[1:] == raster_data.shape[1:]:
    combined_data = np.concatenate((vector_data, raster_data), axis=0)
    print(f"Combined array shape: {combined_data.shape}")

    # Initialize the layer name mapping list
    combined_layer_names = []

    # Add raster layer names (from file names)
    for raster_name in raster_names:
        combined_layer_names.append(raster_name)

    # Add vector layer names (from feature names in vector_feature_grids)
    for vector_feature_name in vector_feature_grids.keys():
        combined_layer_names.append(vector_feature_name)

    # Check the mapping to ensure it is correct
    print("Layer Name Mapping List:", combined_layer_names)

    # Ensure the combined_data layers match the number of names
    if len(combined_layer_names) == combined_data.shape[0]:
        print(f"Layer name mapping successful. Total layers: {len(combined_layer_names)}")
    else:
        print(f"Warning: Mismatch in layers. {len(combined_layer_names)} names for {combined_data.shape[0]} layers.")
else:
    print("Error: The x/y dimensions of the arrays do not match.")


# %%

# PLOT COMBINED DATA WITH INTERACTIVE SLIDER

def plot_combined_data_with_slider(combined_data, combined_layer_names):
    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Initial plot setup (show the first layer)
    layer_index = 0
    current_layer = combined_data[layer_index]
    img = ax.imshow(current_layer, cmap='viridis')
    ax.set_title(f"Layer: {combined_layer_names[layer_index]}")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Define the axes for the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    # Create a slider that allows for layer selection
    slider = Slider(ax_slider, 'Layer', 0, combined_data.shape[0] - 1, valinit=layer_index, valstep=1)

    # Function to update the plot when the slider is changed
    def update(val):
        layer_index = int(slider.val)
        current_layer = combined_data[layer_index]
        img.set_data(current_layer)
        ax.set_title(f"Layer: {combined_layer_names[layer_index]}")
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    # Show the interactive plot
    plt.show()

# Call the function to plot with slider
plot_combined_data_with_slider(combined_data, combined_layer_names)

# Once the plot is closed, the script will continue executing
#%%

# CONVERT TO XARRAY

# Create dummy arrays for X, Y coordinates (you can replace these with your actual coordinates)
x_coords = np.arange(combined_data.shape[2])  # X-coordinates (along the third axis)
y_coords = np.arange(combined_data.shape[1])  # Y-coordinates (along the second axis)
layer_names = combined_layer_names  # Layer names

# Create an xarray DataArray from the combined NumPy array
data_xr = xr.DataArray(
    combined_data, 
    dims=["layer", "y", "x"], 
    coords={"layer": layer_names, "y": y_coords, "x": x_coords},
    name="combined_layers"
)

#%%


#EXPORT XARRAY TO NETCDF

# Hide the root window for the file dialog
root = Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)  # Bring the file dialog to the front

# Prompt the user to select a location to save the NetCDF file
output_file = asksaveasfilename(
    initialfile="THE_CUBE.nc",  # Default file name
    defaultextension=".nc",  # Default extension
    filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
    title="Save NetCDF file"
)

# If the user provides a location, save the NetCDF file
if output_file:
    data_xr.to_netcdf(output_file)
    print(f"Data successfully exported to {output_file}")

# Destroy the root window after file dialog is closed
root.destroy()

