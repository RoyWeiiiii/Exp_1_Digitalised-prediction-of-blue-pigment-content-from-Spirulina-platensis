############# Image cropping and segmentation to remove the gray part ########################
from PIL import Image, ImageStat
import os

# Function to calculate the average pixel color of an image region
def average_color(image, left, upper, right, lower):
    region = image.crop((left, upper, right, lower))
    stat = ImageStat.Stat(region)
    return stat.mean[:3]

# Function to check if an image region has a uniform color
def is_uniform_color(image, left, upper, right, lower, target_color, tolerance=10):
    color = average_color(image, left, upper, right, lower)
    return all(abs(color[i] - target_color[i]) <= tolerance for i in range(3))

# Function to check if the central region of an image has a balanced color
def is_balanced_color(image, target_color, tolerance=10):
    width, height = image.size
    center_left = width // 4
    center_upper = height // 4
    center_right = 3 * (width // 4)
    center_lower = 3 * (height // 4)
    center_color = average_color(image, center_left, center_upper, center_right, center_lower)
    # Check if the central region color is different from the target color
    return not all(abs(center_color[i] - target_color[i]) <= tolerance for i in range(3))

# Function to crop and save regions from an image
def crop_and_save_all_coordinates(image_path, output_folder, crop_size, target_color, tolerance=10):
    # Open the image
    img = Image.open(image_path)

    # Get the size of the image
    width, height = img.size

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the target hex color to RGB
    target_color = tuple(int(target_color[i:i+2], 16) for i in (1, 3, 5))

    # Loop through the image and crop 250x250 regions for every coordinate
    for y in range(0, height - crop_size + 1, crop_size):
        for x in range(0, width - crop_size + 1, crop_size):
            left = x
            upper = y
            right = x + crop_size
            lower = y + crop_size

            # Check if the region has a uniform color that matches the target color
            if not is_uniform_color(img, left, upper, right, lower, target_color, tolerance):
                # Crop the non-uniform color region
                cropped_region = img.crop((left, upper, right, lower))

                # Check if the central region has a balanced color
                if is_balanced_color(cropped_region, target_color, tolerance):
                    # Generate a unique filename for the cropped region
                    output_filename = os.path.join(output_folder, f"{base_filename}_crop_{x}_{y}.jpg")

                    # Save the cropped region
                    cropped_region.save(output_filename)
                    print(f"Saved {output_filename}")

# Function to crop a list of images in a folder
def crop_images_in_folder(input_folder, output_folder, crop_size, target_color, tolerance=10):
    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through the files and crop each image
    for file in files:
        if file.endswith(('.JPG', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(input_folder, file)
            crop_and_save_all_coordinates(input_image_path, output_folder, crop_size, target_color, tolerance)

if __name__ == "__main__":
    # Specify the input folder containing images, the output folder for cropped images,
    # and the fixed crop size (250x250 pixels).
    input_folder = "E:\CodingProjects\machine_learning\Experiment 2\Bg-11\Day_1_BG-11\Batch_1" # Change this to your input folder path
    output_folder = "E:\CodingProjects\machine_learning\Experiment 2\Bg-11\Day_1_BG-11\Batch_1_IMGSEG_Crop_X100" #ange this to your desired output folder
    crop_size = 250

    # Specify the target color in hex format (e.g., #C1C0C5)
    target_color_hex = "#A6A7A9"

    # Call the crop_images_in_folder function
    crop_images_in_folder(input_folder, output_folder, crop_size, target_color_hex)













