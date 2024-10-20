#%%PACKAGE IMPORTS-----------------------------------------------------------------------------------------------------------------------------------------

from typing import Tuple, Dict, Any, List
import os
from tkinter import Tk, filedialog, simpledialog, messagebox, Button, Label
import tkinter as tk
from tkinter.filedialog import askopenfilenames, askdirectory, asksaveasfilename, askopenfilename
from tkinter.simpledialog import askinteger

import geopandas as gpd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import panel as pn
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import xarray as xr
import seaborn as sns

from IPython.display import display

from joblib import Parallel, delayed

from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString


#%%FUNCTIONS-----------------------------------------------------------------------------------------------------------------------------------------------

##File Selection--------------------------------------------------
#File selection function (replace tk with glob in future)
def select_files(title, filetypes, multiple=True):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if multiple:
        file_paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    else:
        file_paths = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return list(file_paths) if multiple else file_paths

# Select all required files in one popup
def select_all_files():
    mask_file = select_files("Select a Mask GeoJSON File", [("GeoJSON files", "*.geojson"), ("All files", "*.*")], multiple=False)
    geojson_files = select_files("Select GeoJSON Files", [("GeoJSON files", "*.geojson"), ("All files", "*.*")])
    line_geojson_files = select_files("Select GeoJSON Line Files", [("GeoJSON files", "*.geojson"), ("All files", "*.*")])
    target_files = select_files("Select Target GeoJSON Files", [("GeoJSON files", "*.geojson"), ("All files", "*.*")])
    raster_files = select_files("Select Raster Files", [("GeoTIFF files", "*.tif"), ("All files", "*.*")])
    return mask_file, geojson_files, line_geojson_files, target_files, raster_files

# Function to select columns from a GeoJSON file
def select_columns(geojson_file):
    gdf = gpd.read_file(geojson_file)
    columns = gdf.columns.tolist()

    # Create a Tkinter window for column selection
    root = tk.Tk()
    root.title(f"Select Columns for {os.path.basename(geojson_file)}")
    root.geometry("400x400")

    selected_columns = []

    # Create checkboxes for each column
    for col in columns:
        var = tk.BooleanVar()
        chk = tk.Checkbutton(root, text=col, variable=var)
        chk.pack(anchor='w')
        selected_columns.append((col, var))

    # Function to handle submission
    def submit():
        selected = [col for col, var in selected_columns if var.get()]
        vector_features_to_process.extend([(geojson_file, col) for col in selected])
        print(f"Selected columns from {geojson_file}: {selected}")
        root.destroy()

    # Submit button
    submit_button = tk.Button(root, text="Submit Selection", command=submit)
    submit_button.pack()

    root.mainloop()

#Grid Size--------------------------------------------------
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

# UNIFIED PROCESSING FUNCTION WITH COMMON GRID ALIGNMENT--------------------------------------------------
def process_file(file, file_type, grid_size, mask_file=None, vector_features_to_process=None, buffer_lines=None, buffer_distance_meters=None):
    # Define grid bounds based on the mask file
    mask_gdf = gpd.read_file(mask_file)
    mask_gdf = mask_gdf.to_crs("EPSG:3857")  # Ensure a projected CRS for mask
    minx, miny, maxx, maxy = mask_gdf.total_bounds

    # Debug step to check the bounds
    #print(f"Grid bounds - minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}")

    # Define the transform for the entire grid area based on the mask bounds
    common_transform = from_bounds(minx, miny, maxx, maxy, grid_size[1], grid_size[0])

    # Debug step to verify function inputs
    #print(f"Buffer lines: {buffer_lines}, Buffer distance meters: {buffer_distance_meters}")

    if file_type == 'target':
        # Buffer and rasterize targets
        target_buffer_size = get_buffer_size()
        target_df = gpd.read_file(file)
        target_df_projected = target_df.to_crs("EPSG:3857")
        target_df_projected['geometry'] = target_df_projected.geometry.buffer(target_buffer_size)
        target_geometry_generator = ((geom, 1) for geom in target_df_projected.geometry)
        target_data = rasterize(shapes=target_geometry_generator, out_shape=grid_size, fill=0, transform=common_transform).astype('float32')
        target_data_3D = np.expand_dims(target_data, axis=0)
        target_layer_name = f"TARGET_{os.path.basename(file).replace('.geojson', '')}"
        return target_data_3D, target_layer_name

    elif file_type == 'vector':
        # Process vector data to create numpy grid
        gdf = gpd.read_file(file)
        gdf = gdf.to_crs("EPSG:3857")
        feature_columns = [col for _, col in vector_features_to_process if _ == file]
        vector_layers = []
        vector_layer_names = []
        for feature_column in feature_columns:
            unique_categories = gdf[feature_column].unique()
            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
            sindex = gdf.sindex
            x = np.linspace(minx, maxx, grid_size[1] + 1)
            y = np.linspace(miny, maxy, grid_size[0] + 1)
            cells = [box(x[j], y[i], x[j + 1], y[i + 1]) for i in range(grid_size[0]) for j in range(grid_size[1])]
            results = Parallel(n_jobs=-1)(delayed(process_cell)(idx, cell, gdf, sindex, feature_column, category_to_int, grid_size) for idx, cell in enumerate(cells))
            grid = np.full(grid_size, np.nan)
            for i, j, value in results:
                grid[i, j] = value
            grid_flipped = np.flipud(grid)
            vector_layers.append(np.expand_dims(grid_flipped, axis=0))
            vector_layer_names.append(f"{os.path.basename(file).replace('.geojson', '')}_{feature_column}")
        return np.concatenate(vector_layers, axis=0), vector_layer_names

    elif file_type == 'line':
        # Process line vector files with buffering and distance calculation
        buffer_lines = user_buffer_choice(file)
        buffer_distance_meters = get_buffer_distance_meters(file) if buffer_lines else 0
        gdf = gpd.read_file(file)
    
        # Reproject to match the CRS of the mask (common grid)
        gdf = gdf.to_crs(mask_gdf.crs)
    
        # Clip the line data to the common bounding box
        common_bounds = box(minx, miny, maxx, maxy)
        gdf_clipped = gdf[gdf.intersects(common_bounds)]
    
        # If there's no intersection, skip this file
        if gdf_clipped.empty:
            print(f"Warning: No valid area for line data within the common grid for file {file}. Skipping...")
            raster_map = np.zeros(grid_size, dtype=np.float32)
            line_layer_name = f"LINE_{os.path.basename(file).replace('.geojson', '')}"
            return np.expand_dims(raster_map, axis=0), line_layer_name
    
        # Rasterize the clipped geometries
        shapes = [(geom, 1) for geom in gdf_clipped.geometry if geom is not None]
        if shapes:
            binary_grid = rasterize(
                shapes=shapes,
                out_shape=grid_size,
                transform=common_transform,
                fill=0,
                dtype='uint8'
            )
    
            # Calculate pixel distance from buffer distance in meters using the common grid resolution
            x_res = common_transform[0]  # Resolution in x-direction (width of a pixel in CRS units)
            y_res = abs(common_transform[4])  # Resolution in y-direction (height of a pixel in CRS units, typically negative)
            
            # Average resolution for the purpose of distance calculation
            avg_resolution = (x_res + y_res) / 2
            pixel_distance = int(buffer_distance_meters / avg_resolution)
    
            # Calculate the distance map if buffer_lines is True
            if buffer_lines and pixel_distance > 0:
                raster_map = calculate_distance(binary_grid, max_distance=pixel_distance)
            else:
                raster_map = binary_grid.astype(np.float32)
        else:
            print(f"No valid line geometries found in {file}. Skipping...")
            raster_map = np.zeros(grid_size, dtype=np.float32)
    
        line_layer_name = f"LINE_{os.path.basename(file).replace('.geojson', '')}"
        return np.expand_dims(raster_map, axis=0), line_layer_name

    
    elif file_type == 'raster':
        # Process raster files
        if not mask_file:
            print("No mask file provided. Skipping raster processing.")
            return None, None
        mask_gdf = gpd.read_file(mask_file)
        minx, miny, maxx, maxy = mask_gdf.total_bounds
        raster_target_transform = from_bounds(minx, miny, maxx, maxy, grid_size[1], grid_size[0])
        raster_target_crs = "EPSG:4326"
        with rasterio.open(file, 'r') as src:
            print(f"Processing file: {file}")
            src_crs = src.crs if src.crs != raster_target_crs else raster_target_crs
            raster_data_array = np.full(grid_size, np.nan, dtype=np.float32)
            nodata_value = src.nodata if src.nodata is not None else np.nan
            reproject(
                source=rasterio.band(src, 1),
                destination=raster_data_array,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=common_transform,
                dst_crs=raster_target_crs,
                resampling=Resampling.nearest,
                src_nodata=nodata_value,
                dst_nodata=np.nan
            )
        raster_name = os.path.basename(file).replace('.tiff', '').replace('.tif', '')
        return np.expand_dims(raster_data_array, axis=0), raster_name

    else:
        raise ValueError("Unsupported file type provided.")

# Helper functions
def get_buffer_size():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    buffer_size = simpledialog.askinteger("Input", "Enter the target buffer size in meters:", minvalue=1)
    root.destroy()
    return buffer_size

def process_cell(idx, cell, gdf, sindex, feature_column, category_to_int, grid_size):
    i, j = divmod(idx, grid_size[1])
    possible_matches_index = list(sindex.intersection(cell.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    intersections = possible_matches.geometry.intersection(cell)
    valid_intersections = intersections[intersections.area > 0]
    if valid_intersections.empty:
        return i, j, np.nan
    largest_intersection_idx = valid_intersections.area.idxmax()
    category = possible_matches.loc[largest_intersection_idx, feature_column]
    return i, j, category_to_int[category]

def calculate_distance(arr, max_distance=20, dtype=np.float32):
    arr = np.asarray(arr, dtype=bool)
    dist = distance_transform_edt(~arr)
    normalized_dist = np.clip(1 - dist / max_distance, 0, 1)
    return normalized_dist.astype(dtype)

# Function to calculate pixel distance from buffer distance in meters
def calculate_pixel_distance(gdf, grid_size, buffer_distance_meters):
    height, width = grid_size
    bounds = gdf.total_bounds
    x_range_meters = bounds[2] - bounds[0]
    y_range_meters = bounds[3] - bounds[1]
    x_res = x_range_meters / width
    y_res = y_range_meters / height
    pixel_distance = buffer_distance_meters / ((x_res + y_res) / 2)
    return int(pixel_distance)

# Function for user to choose between buffering or rasterizing
def user_buffer_choice(file):
    choice = {"buffer": None}

    def set_choice_buffer():
        choice["buffer"] = True
        window.destroy()

    def set_choice_rasterize():
        choice["buffer"] = False
        window.destroy()

    window = Tk()
    window.title("Choose Processing Option")
    label = Label(window, text=f"Would you like to buffer the lines (calculate distances) or just rasterize them for {os.path.basename(file)}?")
    label.pack(pady=10)
    buffer_button = Button(window, text="Buffer (Calculate Distance)", command=set_choice_buffer)
    buffer_button.pack(side="left", padx=20, pady=20)
    rasterize_button = Button(window, text="Rasterize Only", command=set_choice_rasterize)
    rasterize_button.pack(side="right", padx=20, pady=20)
    window.mainloop()
    return choice["buffer"]

def get_buffer_distance_meters(file):
    root = Tk()
    root.withdraw()
    buffer_distance = None
    try:
        buffer_distance = simpledialog.askfloat(
            title=f"Input Buffer Distance for {os.path.basename(file)}",
            prompt="Please enter buffer distance in meters:"
        )
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number.")
        buffer_distance = None
    root.destroy()
    return buffer_distance


#%%FILE IMPORT-----------------------------------------------------------------------------------------------------------------------------------------

# Use the function to select all files
mask_file, geojson_files, line_geojson_files, target_files, raster_files = select_all_files()

# List to store selected columns from GeoJSON files
vector_features_to_process = []

# Loop through GeoJSON files and select columns
for file in geojson_files:
    select_columns(file)


#%%GRID SIZE-----------------------------------------------------------------------------------------------------------------------------------------

# Prompt the user for the short_edge_cells value using tkinter
root = Tk()
root.withdraw()  # Hide the root window
root.attributes("-topmost", True)  # Ensure it is on top


# Ask the user for the short edge size
short_edge_cells = simpledialog.askinteger("Input", "Enter the number of cells for the short edge:", minvalue=1)

root.destroy()  # Close the tkinter root window

if short_edge_cells is None:
    raise ValueError("You must enter a valid number for the short edge size.")

# Compute grid size using the mask file
grid_size = compute_grid_size(mask_file, short_edge_cells=short_edge_cells)[::-1]
print(f"Calculated grid size: {grid_size}")


#%%PROCESS LAYERS-----------------------------------------------------------------------------------------------------------------------------------------

# Process targets
target_data = []
target_layer_names = []
if target_files:
    for file in target_files:
        target_layer, target_layer_name = process_file(file, 'target', grid_size, mask_file=mask_file)
        target_data.append(target_layer)
        target_layer_names.append(target_layer_name)
    target_data = np.concatenate(target_data, axis=0)
    print(f"Target data shape: {target_data.shape}")
else:
    print("No target files detected.")
    target_data = np.array([])
    target_layer_names = []

# Process rasters
raster_data = []
raster_names = []
if raster_files:
    if not mask_file:
        print("No mask file selected. Skipping raster processing.")
    else:
        for file in raster_files:
            raster_layer, raster_name = process_file(file, 'raster', grid_size, mask_file=mask_file)
            if raster_layer is not None:
                raster_data.append(raster_layer)
                raster_names.append(raster_name)
        if raster_data:
            raster_data = np.concatenate(raster_data, axis=0)
            print(f"Raster data shape: {raster_data.shape}")
        else:
            print("No raster data processed.")
else:
    print("No raster files detected.")
    raster_data = np.array([])
    raster_names = []

# Process line vectors
line_vector_data = []
line_layer_names = []
if line_geojson_files:
    for file in line_geojson_files:
        line_layer, line_layer_name = process_file(file, 'line', grid_size, mask_file=mask_file)
        line_vector_data.append(line_layer)
        line_layer_names.append(line_layer_name)
    line_vector_data = np.concatenate(line_vector_data, axis=0)
    print(f"Line vector data shape: {line_vector_data.shape}")
else:
    print("No line vector files selected.")
    line_vector_data = np.array([])
    line_layer_names = []

# Process polygon vectors
vector_data = []
vector_feature_grids = {}
vector_layer_names = []
if geojson_files and vector_features_to_process:
    for file in geojson_files:
        vector_layer, feature_names = process_file(file, 'vector', grid_size, mask_file=mask_file, vector_features_to_process=vector_features_to_process)
        vector_data.append(vector_layer)
        vector_layer_names.extend(feature_names)
        for feature_name, layer in zip(feature_names, vector_layer):
            vector_feature_grids[feature_name] = layer
    vector_data = np.concatenate(vector_data, axis=0)
    print(f"Vector data shape: {vector_data.shape}")
else:
    print("No polygon vector files or features selected.")
    vector_data = np.array([])
    vector_feature_grids = {}
    vector_layer_names = []

print(f"vector_data shape: {vector_data.shape}")
print(f"line_vector_data shape: {line_vector_data.shape}")
print(f"raster_data shape: {raster_data.shape}")
print(f"target_data shape: {target_data.shape}")


#%%COMBINE LAYERS-----------------------------------------------------------------------------------------------------------------------------------------

combined_data = []
combined_layer_names = []

# Create a list to hold non-empty data arrays and their corresponding layer names
data_arrays = []
layer_names = []

# Function to add non-empty data arrays and their names
def add_data_and_names(data_array, names):
    if data_array.size != 0:
        data_arrays.append(data_array)
        layer_names.extend(names)
    else:
        print(f"Skipping empty data array with names: {names}")

# Add vector data
if vector_data.size != 0:
    add_data_and_names(vector_data, vector_layer_names)
else:
    print("Vector data is empty.")

# Add raster data
if raster_data.size != 0:
    add_data_and_names(raster_data, raster_names)
else:
    print("Raster data is empty.")

# Add line vector data
if line_vector_data.size != 0:
    add_data_and_names(line_vector_data, line_layer_names)
else:
    print("Line vector data is empty.")

# Add target data
if target_data.size != 0:
    add_data_and_names(target_data, target_layer_names)
else:
    print("Target data is empty.")

# Check if we have at least one non-empty data array
if data_arrays:
    # Check that all arrays have the same x and y dimensions
    first_shape = data_arrays[0].shape[1:]  # Skip the first dimension (number of layers)
    shapes_match = all(data_array.shape[1:] == first_shape for data_array in data_arrays)

    if shapes_match:
        # Concatenate all data arrays along axis=0
        combined_data = np.concatenate(data_arrays, axis=0)
        combined_layer_names = layer_names

        print(f"Combined array shape (with targets): {combined_data.shape}")

        # Check the mapping to ensure it is correct
        print("Layer Name Mapping List:", combined_layer_names)

        # Ensure the combined_data layers match the number of names
        if len(combined_layer_names) == combined_data.shape[0]:
            print(f"Layer name mapping successful. Total layers: {len(combined_layer_names)}")
        else:
            print(f"Warning: Mismatch in layers. {len(combined_layer_names)} names for {combined_data.shape[0]} layers.")
    else:
        print("Error: The x/y dimensions of the arrays do not match.")
        print("Array shapes:")
        for idx, data_array in enumerate(data_arrays):
            print(f"Array {idx} shape: {data_array.shape}")
else:
    print("No data arrays to combine.")

#%%CONVERT TO XARRAY/EXPORT-----------------------------------------------------------------------------------------------------------------------------------------

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