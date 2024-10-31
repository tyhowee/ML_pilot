import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename

# Prompt the user to select input files----------------------------------------------------------------------
root = Tk()
root.withdraw()  # Hide the main window
root.attributes("-topmost", True)  # Bring the dialog to the front
file_paths = askopenfilenames(
    title="Select TIFF files to average",
    filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
)
root.update()  # Update to make sure the selection goes through

if not file_paths:
    print("No files selected. Exiting.")
    exit()

# Load all files into a 3D array to calculate pixel-by-pixel statistics
data_stack = []

for file_path in file_paths:
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first (or only) band of the TIFF
        data_stack.append(data)

# Convert the list of arrays to a 3D numpy array (stacked along the third dimension)
data_stack = np.stack(data_stack, axis=2)

# Calculate pixel-by-pixel statistics
average_array = np.mean(data_stack, axis=2)
min_array = np.min(data_stack, axis=2)
max_array = np.max(data_stack, axis=2)
std_dev_array = np.std(data_stack, axis=2)

# Prompt the user to select the output file path and name
output_file = asksaveasfilename(
    title="Save Averaged Probability Map as TIFF",
    defaultextension=".tif",
    filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
)

if not output_file:
    print("No output file selected. Exiting.")
    exit()

# Define output directory and base filename
output_dir = os.path.dirname(output_file)
base_name = os.path.splitext(os.path.basename(output_file))[0]

# Save pixel-by-pixel statistics as separate TIFF files
def save_array_as_tiff(array, filename, template_file):
    with rasterio.open(template_file) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float64)

        with rasterio.open(filename, "w", **profile) as dst:
            dst.write(array, 1)

# Save averaged probability map as TIFF
save_array_as_tiff(average_array, output_file, file_paths[0])
print(f"Averaged probability map saved as '{output_file}'")

# Save min, max, and std deviation as separate TIFF files
min_tiff_path = os.path.join(output_dir, f"{base_name}_min.tif")
max_tiff_path = os.path.join(output_dir, f"{base_name}_max.tif")
std_dev_tiff_path = os.path.join(output_dir, f"{base_name}_std_dev.tif")

save_array_as_tiff(min_array, min_tiff_path, file_paths[0])
save_array_as_tiff(max_array, max_tiff_path, file_paths[0])
save_array_as_tiff(std_dev_array, std_dev_tiff_path, file_paths[0])

print(f"Min map saved as '{min_tiff_path}'")
print(f"Max map saved as '{max_tiff_path}'")
print(f"Standard deviation map saved as '{std_dev_tiff_path}'")


# Plot the averaged probability map----------------------------------------------------------------------
plt.figure(figsize=(14, 10))
plt.imshow(average_array, cmap='viridis')
plt.colorbar(label='Probability')
plt.title('Mineral Deposit Probability Map (AVERAGED)', fontsize=15)
plt.axis('off')
plt.tight_layout()

# Save the plot as a PNG file with the same name as the output TIFF
png_file_path = output_file.replace(".tif", ".png")
plt.savefig(png_file_path, format='png')
print(f"Probability map image saved as '{png_file_path}'")

plt.show()
