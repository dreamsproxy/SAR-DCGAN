import os
import glob
import shutil
import imageio

# Source directory containing the images
source_dir = 'datasets/SAI/gray_256/'

# Destination directory where to move modified images
dest_dir = 'datasets/SAI/bad/'

# Set the brightness threshold for reducing the brightness
brightness_threshold = 0.18

# Get a list of all grayscale images in the source directory
image_files = glob.glob(os.path.join(source_dir, '**.png'))

for image_file in image_files:
    # Load the grayscale image using imageio
    img = imageio.imread(image_file, as_gray=True)

    # Compute the normalized brightness between 0 and 1
    brightness = img.mean()/255

    # Check if the brightness is above the threshold
    if brightness > brightness_threshold:
        # Reduce the brightness by 50%
        #img = img * 0.5

        # Save the modified image to the destination directory
        file_name = os.path.basename(image_file)
        dest_file = os.path.join(dest_dir, file_name)
        imageio.imwrite(dest_file, img)

        # Remove the original image from the source directory
        os.remove(image_file)
