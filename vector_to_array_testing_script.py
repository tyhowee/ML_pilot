# IMPORTS
import cupy as cp
import cudf
import numpy
from cuspatial import cuspatial
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from typing import Tuple, Dict, Any, List
from tkinter import Tk, Toplevel, Button, Checkbutton, IntVar, Label, filedialog, simpledialog

# RAPIDS GPU-Accelerated Processing Function

def process_chunk_gpu(chunk, gdf, sindex, feature_column, category_to_int, filename_prefix, grid_size):
    results = []
    for idx, cell in chunk:
        i, j = divmod(idx, grid_size[1])
        possible_matches_index = list(sindex.query(cell.bounds))
        
        if not possible_matches_index:
            # No intersecting features, return NaN for no data
            results.append((i, j, cp.nan))
            continue

        # Check for actual intersection and assign the feature value
        possible_matches = gdf.iloc[possible_matches_index]
        if possible_matches.empty:
            results.append((i, j, cp.nan))
            continue

        # Calculate intersections more precisely
        intersections = cuspatial.intersection(possible_matches.geometry, cell)

        # Consider all non-zero intersections
        valid_intersections = intersections[intersections.area > 0]

        if valid_intersections.empty:
            results.append((i, j, cp.nan))
            continue

        # Determine strategy based on the number of intersecting features
        if len(valid_intersections) > 5:  # Threshold for choosing strategy
            # Many small polygons: Sum the areas for each unique category
            areas_per_category = {}
            for idx, intersection in enumerate(valid_intersections):
                if not intersection.is_empty:
                    category = possible_matches.iloc[idx][feature_column]
                    category_key = f"{filename_prefix}_{category}"
                    if category_key not in areas_per_category:
                        areas_per_category[category_key] = 0
                    areas_per_category[category_key] += intersection.area

            # Choose the category with the largest cumulative area
            if areas_per_category:
                max_category = max(areas_per_category, key=areas_per_category.get)
                results.append((i, j, category_to_int[max_category]))
            else:
                results.append((i, j, cp.nan))  # Fallback to NaN

        else:
            # Few large polygons: Choose the largest single intersection by area
            largest_intersection_idx = valid_intersections.area.idxmax()
            category = possible_matches.loc[largest_intersection_idx, feature_column]
            results.append((i, j, category_to_int[f"{filename_prefix}_{category}"]))
    return results

# Function to process each feature column using GPU
def process_feature_column_gpu(gdf, feature_column, grid_size, target_crs, filename_prefix, x, y):
    gdf = gdf.to_crs(target_crs)
    unique_categories = gdf[feature_column].unique()
    category_to_int = {f"{filename_prefix}_{cat}": i for i, cat in enumerate(unique_categories)}

    grid = cp.full(grid_size, cp.nan)
    sindex = gdf.sindex  # RAPIDS Spatial Index

    cells = [(idx, box(x[j], y[i], x[j + 1], y[i + 1]))
             for idx, (i, j) in enumerate(cp.ndindex(grid_size))]

    chunk_size = max(1, len(cells) // 10)
    chunks = [cells[i:i + chunk_size] for i in range(0, len(cells), chunk_size)]

    # GPU-parallel processing using RAPIDS
    results = [process_chunk_gpu(
        chunk, gdf, sindex, feature_column, category_to_int, filename_prefix, grid_size
    ) for chunk in chunks]

    for result in results:
        for i, j, value in result:
            grid[i, j] = value

    return (f"{filename_prefix}_{feature_column}", grid, category_to_int)

# Main Function using RAPIDS GPU Acceleration
def geojson_to_numpy_grid_3d_batch_gpu(
    grid_size: Tuple[int, int],  # Grid size for the output array
    target_crs: str = "EPSG:3857"  # Web Mercator projection
) -> Tuple[cp.ndarray, Dict[str, cp.ndarray], Dict[str, Dict[Any, int]], List[Dict[str, Any]]]:
    # Use tkinter to select files
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    geojson_files = filedialog.askopenfilenames(
        title="Select GeoJSON Files",
        filetypes=[("GeoJSON files", "*.geojson"), ("All files", "*.*")]
    )
    root.destroy()

    if not geojson_files:
        raise ValueError("No files selected. Please select at least one GeoJSON file.")

    # Load all selected files into cuDF DataFrames
    gdf_dict = {geojson_file: cudf.read_file(geojson_file) for geojson_file in geojson_files}

    # Ask the user to select features to process
    selected_features = select_features(gdf_dict)

    all_feature_grids = {}
    all_feature_mappings = {}
    geospatial_info_list = []

    results = []

    for geojson_file in geojson_files:
        gdf = gdf_dict[geojson_file]
        gdf = gdf.to_crs(target_crs)
        minx, miny, maxx, maxy = gdf.total_bounds

        x = cp.linspace(minx, maxx, grid_size[1] + 1)
        y = cp.linspace(miny, maxy, grid_size[0] + 1)

        feature_columns = selected_features.get(geojson_file, [])

        if not feature_columns:
            print(f"No features selected for file: {geojson_file}")
            continue

        filename_prefix = os.path.splitext(os.path.basename(geojson_file))[0]

        geospatial_info = {
            'transform': (minx, miny, maxx, maxy),
            'crs': target_crs,
            'file_name': filename_prefix
        }
        geospatial_info_list.append(geospatial_info)

        for feature_column in feature_columns:
            result = process_feature_column_gpu(
                gdf, feature_column, grid_size, target_crs, filename_prefix, x, y
            )
            results.append(result)

    for feature_name, grid, category_to_int in results:
        all_feature_grids[feature_name] = grid
        all_feature_mappings[feature_name] = category_to_int

    if not all_feature_grids:
        raise ValueError("No features were processed. Please select at least one feature to process.")

    grid_3d = cp.stack(list(all_feature_grids.values()), axis=0)

    return grid_3d, all_feature_grids, all_feature_mappings, geospatial_info_list

#%%

# COMPUTE GRID SIZE

def compute_grid_size(geojson_file: str, short_edge_cells: int = 1200) -> Tuple[int, int]:
    # Read the GeoJSON file
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


mask_file = r"C:\Users\TyHow\Documents\3. Work\GIS Stuff\ML_pilot_data\MASK.geojson"
grid_size = compute_grid_size(mask_file, short_edge_cells=short_edge_cells)[::-1]
print(f"Calculated grid size: {grid_size}")

# %%
#RUN FUNCTION

#grid_size = (1200, 1550)  # Define the grid size

# Call the function
grid_3d, feature_grids, feature_mappings, geospatial_info_list = geojson_to_numpy_grid_3d_batch(grid_size)

# Print results
print("Shape of the 3D grid array:", grid_3d.shape)
print("Feature grids:", feature_grids.keys())
print("Feature mappings:", feature_mappings)
print("Geospatial information for each file:", geospatial_info_list)

#%%


def plot_layers_with_slider(feature_grids, grid_3d):
    """
    Create an interactive plot window with a slider to view various layers using Matplotlib.
    """
    # Create the initial plot
    fig, ax = plt.subplots(figsize=(6, 8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom space for the slider

    # Initial layer index
    layer_index = 0

    # Determine the global min and max values across all layers for consistent color scaling
    global_min = np.nanmin(grid_3d)
    global_max = np.nanmax(grid_3d)

    # Plot the initial layer with consistent color normalization
    im = ax.imshow(grid_3d[layer_index], cmap='tab20', interpolation='nearest', aspect='auto',
                   vmin=global_min, vmax=global_max)
    title = ax.set_title(f"Layer {layer_index + 1}: {list(feature_grids.keys())[layer_index]}")
    plt.colorbar(im, ax=ax, label='Classes')

    # Create an axis for the slider
    slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')

    # Create the slider
    layer_slider = Slider(slider_ax, 'Layer Index', 0, grid_3d.shape[0] - 1, valinit=0, valstep=1)

    # Function to update the plot based on slider value
    def update(val):
        layer = int(layer_slider.val)
        im.set_data(grid_3d[layer])
        title.set_text(f"Layer {layer + 1}: {list(feature_grids.keys())[layer]}")
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    layer_slider.on_changed(update)

    # Show the plot
    plt.show()

# Call the visualization function
plot_layers_with_slider(feature_grids, grid_3d)



