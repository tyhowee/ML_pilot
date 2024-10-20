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



#%%FILE IMPORT----------------------------------------------------------------------------

# Use the function to select all files
mask_file, geojson_files, line_geojson_files, target_files, raster_files = select_all_files()

# List to store selected columns from GeoJSON files
vector_features_to_process = []

# Loop through GeoJSON files and select columns
for file in geojson_files:
    select_columns(file)


#%%GRID SIZE----------------------------------------------------------------------------

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
