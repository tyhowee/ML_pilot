import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
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

# Initialize an array to store the sum of the selected TIFF files
sum_array = None

# Loop over each selected file and add it to the sum
for file_path in file_paths:
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first (or only) band of the TIFF
        
        # Initialize sum_array with the first file's shape and dtype if not done yet
        if sum_array is None:
            sum_array = np.zeros_like(data, dtype=np.float64)  # Use float64 for cumulative sum accuracy

        # Add the current file's data to the sum
        sum_array += data

# Divide the summed array by the number of selected files to get the average
average_array = sum_array / len(file_paths)

# Calculate statistics for the averaged array
min_value = np.min(average_array)
max_value = np.max(average_array)
std_dev = np.std(average_array)

# Prompt the user to select the output file path and name
output_file = asksaveasfilename(
    title="Save Averaged Probability Map as TIFF",
    defaultextension=".tif",
    filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
)

if not output_file:
    print("No output file selected. Exiting.")
    exit()

# Save the statistics in the same directory as the output TIFF file
output_dir = os.path.dirname(output_file)
stats_file_path = os.path.join(output_dir, "prob_stats.txt")

with open(stats_file_path, "w") as stats_file:
    stats_file.write(f"Min Value: {min_value}\n")
    stats_file.write(f"Max Value: {max_value}\n")
    stats_file.write(f"Standard Deviation: {std_dev}\n")

print(f"Probability map statistics saved to '{stats_file_path}'")

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

# Save the averaged array as a new TIFF
with rasterio.open(
    file_paths[0]  # Use the first file's metadata as a template
) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float64)

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(average_array, 1)  # Write the result to the first band

print(f"Averaged result saved as '{output_file}'")
