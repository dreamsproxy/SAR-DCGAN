import cv2
import os
from tqdm import tqdm


# Define the input and output directories
input_dir = './MEDISAR'
output_dir = './GRID-MEDISAR'

# Define the size of the cropped images
crop_size = 512

# Loop through all the files in the input directory
for filename in tqdm(os.listdir(input_dir)):
    # Load the image
    img = cv2.imread(os.path.join(input_dir, filename))
    # Get the height and width of the image
    height, width, _ = img.shape
    # Calculate the number of rows and columns needed to create a grid of square images
    num_rows = int(height / crop_size)
    num_cols = int(width / crop_size)
    # Loop through each row and column of the grid
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates for the top-left and bottom-right corners of the current square
            x1 = col * crop_size
            y1 = row * crop_size
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            # Crop the image and save it to the output directory with a unique filename
            cropped_img = img[y1:y2, x1:x2]
            output_filename = os.path.splitext(filename)[0] + f'_{row}_{col}.jpg'
            cv2.imwrite(os.path.join(output_dir, output_filename), cropped_img)
