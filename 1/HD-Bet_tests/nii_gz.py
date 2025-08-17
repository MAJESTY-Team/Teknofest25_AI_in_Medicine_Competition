from PIL import Image
import SimpleITK as sitk
import numpy as np
import os
import glob

# Get all PNG files from the HD-Bet_tests folder
image_folder = './HD-Bet_tests'
png_files = glob.glob(os.path.join(image_folder, '*.png'))

# Sort the files to ensure consistent ordering
png_files.sort()

print(f"Found {len(png_files)} PNG files")

# Load all images and store them in a list
image_slices = []
for png_file in png_files:
    # Load and convert each image to grayscale
    png_image = Image.open(png_file).convert('L')
    image_array = np.array(png_image)
    image_slices.append(image_array)
    print(f"Loaded: {os.path.basename(png_file)} - Shape: {image_array.shape}")

# Stack all images to create a 3D volume (slices, height, width)
if image_slices:
    image_3d = np.stack(image_slices, axis=0)
    print(f"3D volume shape: {image_3d.shape}")
    
    # Convert the NumPy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_3d)
    
    # Set proper spacing (you may want to adjust these values based on your data)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))  # (x, y, z) spacing in mm
    
    # Save the image as a NIfTI file
    output_filename = 'HD_Bet_tests_3D.nii.gz'
    sitk.WriteImage(sitk_image, output_filename)
    print(f"Successfully created 3D NIfTI file: {output_filename}")
else:
    print("No PNG files found in the specified folder!")





# from PIL import Image
# import SimpleITK as sitk
# import numpy as np

# # Load your PNG image
# png_image = Image.open('HD-Bet_tests - Kopya/58 (30).png').convert('L')  # Convert to grayscale

# # Convert the image to a NumPy array
# image_array = np.array(png_image)

# # Create a 3D image by adding a third dimension
# image_3d = image_array[np.newaxis, :, :]

# # Convert the NumPy array to a SimpleITK image
# sitk_image = sitk.GetImageFromArray(image_3d)

# # Save the image as a NIfTI file
# sitk.WriteImage(sitk_image, 'output.nii.gz')