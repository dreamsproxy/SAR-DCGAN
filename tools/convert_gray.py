import cv2
import os
from tqdm import tqdm

# Define the input and output directories
input_dir = './ships-aerial-images/images'
output_dir = './SAI'

# Define the contrast adjustment parameters
alpha = 1.1  # Contrast control (1.0-3.0)
beta = 0     # Brightness control (0-100)

# Loop through all the files in the input directory
for filename in tqdm(os.listdir(input_dir)):
    # Load the image
    img = cv2.imread(os.path.join(input_dir, filename))
    # Increase the contrast of the image
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Save the processed image to the output directory with a new filename
    output_filename = os.path.splitext(filename)[0] + '_processed.jpg'
    cv2.imwrite(os.path.join(output_dir, output_filename), gray_img)
