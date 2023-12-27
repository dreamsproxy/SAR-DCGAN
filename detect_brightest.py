from PIL import Image, ImageDraw
import heapq
import math
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_brightest_pixels(image, num_pixels, min_distance):
    width, height = image.size
    brightest_pixels = []
    pixels_heap = []

    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            heapq.heappush(pixels_heap, (-pixel_value, (x, y)))

    while pixels_heap and len(brightest_pixels) < num_pixels:
        brightest_pixel = heapq.heappop(pixels_heap)
        pixel_position = brightest_pixel[1]

        # Check distance from other brightest pixels
        is_far_enough = True
        for existing_pixel_position in brightest_pixels:
            distance = math.sqrt((pixel_position[0] - existing_pixel_position[0]) ** 2 + (pixel_position[1] - existing_pixel_position[1]) ** 2)
            if distance < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            brightest_pixels.append(pixel_position)

    return brightest_pixels


def convert_image_to_grayscale(image):
    return image.convert("L")

def normalize_pixel_value(value, min_value, max_value):
    return ((value - min_value) / (max_value - min_value)) * 99 + 1

def find_pixels_in_range(image, lower_range, upper_range):
    width, height = image.size
    pixels_in_range = []

    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            normalized_value = normalize_pixel_value(pixel_value, 0, 255)

            if lower_range <= normalized_value <= upper_range:
                pixels_in_range.append((x, y))

    return pixels_in_range

def find_brightest_pixel(image):
    width, height = image.size
    brightest_pixel_value = 0
    brightest_pixel_position = (0, 0)

    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            if pixel_value > brightest_pixel_value:
                brightest_pixel_value = pixel_value
                brightest_pixel_position = (x, y)

    return brightest_pixel_position

def draw_box_around_pixel(image, pixel_position, box_size, outline_color):
    draw = ImageDraw.Draw(image)
    x, y = pixel_position
    half_box_size = box_size // 2
    box_coordinates = (
        x - half_box_size,
        y - half_box_size,
        x + half_box_size,
        y + half_box_size
    )
    draw.rectangle(box_coordinates, outline=outline_color)
    #print(box_coordinates)
    #print("\n")
    return box_coordinates

def process_image(image_path, lower_range, upper_range, num_brightest_pixels, min_distance, box_size, outline_color):
    image = Image.open(image_path).resize((256,256))
    grayscale_image = convert_image_to_grayscale(image)
    pixels_in_range = find_pixels_in_range(grayscale_image, lower_range, upper_range)
    brightest_pixels = find_brightest_pixels(grayscale_image, num_brightest_pixels, min_distance)
    
    # Draw boxes around the brightest pixels
    for pixel_position in brightest_pixels:
        draw_box_around_pixel(image, pixel_position, box_size, outline_color)
    
    return image

def create_image_grid(images, grid_size):
    num_images = len(images)
    num_rows, num_cols = grid_size

    # Create a figure and set the grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Iterate through the images and plot them in the grid
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis("off")

    # Hide any unused subplots
    for i in range(num_images, num_rows * num_cols):
        axes.flat[i].axis("off")

    # Display the grid plot
    plt.show()

image_log = []
image_list = glob("./datasets/SAI/base_model/**")
for i, f in tqdm(enumerate(image_list), total = len(image_list)):
    # Example usage
    image_path = f

    lower_range = 80
    upper_range = 100
    num_brightest_pixels = 5
    box_size = 10
    outline_color = "white"

    min_distance = 32
    output_image = process_image(image_path, lower_range, upper_range, num_brightest_pixels, min_distance, box_size, outline_color)
    if i % 10 == 0:
        image_log.append(output_image)


print(len(image_log))
user_tuple = int(input("grid shape pls >>> "))
create_image_grid(image_log, (user_tuple, user_tuple))
#output_image.show()