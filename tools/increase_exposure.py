import os
import imageio
from multiprocessing import Pool

def increase_exposure(filename):
    # Load image
    image = imageio.imread(filename)

    # Increase exposure by 100%
    image = image * 2.5

    # Save image with "_exposed" suffix
    new_filename = os.path.splitext(filename)[0] + '_exposed' + os.path.splitext(filename)[1]
    imageio.imwrite(new_filename, image)

if __name__ == '__main__':
    # Get list of all image files in directory
    directory = './GRID-MEDISAR'
    image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Create pool of worker processes
    pool = Pool()

    # Apply increase_exposure function to each image in parallel
    pool.map(increase_exposure, image_files)

    # Close pool
    pool.close()
    pool.join()
