import os
import numpy as np
import imageio
import multiprocessing
from tqdm import tqdm

def make_square_image(input_file_path):
    #print(type(input_file_path))
    output_dir = input_file_path[1]
    input_file_path = input_file_path[0]
    """
    Reads the input image and copies it to a black background with a 1:1 aspect ratio,
    then saves the result to the specified output path
    """
    
    # Load the input image
    image = imageio.imread(input_file_path)

    # Force conversion from rgb to grayscale
    #r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    #image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Determine the new size of the image after making it square
    new_size = max(height, width)

    # Create a new array of the correct size filled with zeros (black)
    new_image = np.zeros((new_size, new_size), dtype=np.uint8)

    # Calculate the x,y coordinates to paste the input image onto the new background
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2

    # Paste the input image onto the new background
    new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

    # Save the result to the output directory with the same filename as the input image
    output_file_path = os.path.join(output_dir, os.path.basename(input_file_path))
    output_file_path = output_file_path[:-3] + ".png"
    imageio.imwrite(output_file_path, new_image)

if __name__ == '__main__':
    # Specify the input and output directories
    input_dir = 'datasets/SAI/gray-default_size'
    output_dir = 'datasets/SAI/gray_1-1_ratio'

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all the image files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # Define the number of processes to use (default is the number of CPU cores)
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    num_processes = 6

    progbar = tqdm(total = len(input_files))
    # Use multiprocessing.imap to convert the images in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a generator of the input and output paths for each image file
        file_path_generator = ((f.replace('\\', '/'), output_dir) for f in input_files)

        # Apply the make_square_image function to each image file in parallel
        for result in pool.imap_unordered(make_square_image, file_path_generator):
            progbar.update(1)
            pass
    progbar.close()
