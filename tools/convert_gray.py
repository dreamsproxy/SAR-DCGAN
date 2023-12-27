import os
import imageio
import multiprocessing
from tqdm import tqdm

def grayscale_image(input_file_path, output_file_path):
    """
    Converts the input image to true grayscale and saves it to the specified output path
    """
    image = imageio.imread(input_file_path)
    gray_image = image.mean(axis=2).astype('uint8')
    imageio.imwrite(output_file_path, gray_image)

if __name__ == '__main__':
    # Specify the input and output directories
    input_dir = "datasets/SAI/gray_256/good"
    output_dir = "datasets/used"

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all the image files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # Define the number of processes to use (default is the number of CPU cores)
    num_processes = multiprocessing.cpu_count()

    # Use multiprocessing to convert the images in parallel
    pbar = tqdm(total=len(input_files))
    pool = multiprocessing.Pool(processes=num_processes)
    for input_file_path in tqdm(input_files):
        #print(input_file_path)
        output_file_path = os.path.join(output_dir, os.path.basename(input_file_path))
        pool.apply_async(grayscale_image, args=(input_file_path, output_file_path))
        pbar.update(1)
    pbar.close()
    pool.close()
    pool.join()
