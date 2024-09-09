# import cv2

# # Load the image
# image = cv2.imread("E:\CodingProjects\machine_learning\Experiment 2\Bg-11\Day_1_BG-11\Batch_3\DSC_7281.JPG")

# # Define the desired width and height for resizing
# desired_width = 1000  # Adjust to your preferred width
# desired_height = 1000  # Adjust to your preferred height

# # Resize the image while maintaining its aspect ratio
# resized_image = cv2.resize(image, (desired_width, desired_height))

# # Convert the resized image to grayscale
# gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# # Global thresholding with a fixed threshold_value
# threshold_value = 165  # Adjust this value based on your image
# _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# # Invert the binary mask
# inverted_mask = cv2.bitwise_not(binary_mask)

# # Create a 3-channel color mask
# color_mask = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

# # Use the color mask to extract the object
# object_retained_color = cv2.bitwise_and(resized_image, color_mask)

# # Display or save the resized object with the background removed while retaining its color
# cv2.imshow('Resized Object with Background Removed', object_retained_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################################

# import cv2

# Load the image
# image = cv2.imread("E:\CodingProjects\machine_learning\Experiment 2\Zarrouk\Day_4_Zarrouk\Batch_2\DSC_3092.JPG")

# # Define the desired width and height for resizing
# desired_width = 1000  # Adjust to your preferred width
# desired_height = 1000  # Adjust to your preferred height

# # Resize the image while maintaining its aspect ratio
# resized_image = cv2.resize(image, (desired_width, desired_height))

# # Convert the resized image to grayscale
# gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# # Manual thresholding with a fixed threshold_value
# threshold_value = 170  # Adjust this value based on your image
# _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# # Invert the binary mask
# inverted_mask = cv2.bitwise_not(binary_mask)

# # Create a 3-channel color mask
# color_mask = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

# # Use the color mask to extract the object
# object_retained_color = cv2.bitwise_and(resized_image, color_mask)

# # Display or save the resized object with the background removed while retaining its color
# cv2.imshow('Resized Object with Background Removed (Manual Thresholding)', object_retained_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################### Working for multiple background removal ################
# import cv2
# import os
# import glob

# # Define the source folder containing JPG files
# source_folder = "E:\CodingProjects\machine_learning\Experiment 2\Zarrouk\Day_14_Zarrouk\Batch_1"

# # Define the destination folder to save the results
# destination_folder = "E:\CodingProjects\machine_learning\Experiment 2\Zarrouk\Day_14_Zarrouk\Batch_1_BR"

# # Define the desired width and height for resizing
# desired_width = 3712  # Adjust to your preferred width
# desired_height = 3712  # Adjust to your preferred height

# # Function to perform background removal
# def remove_background(image_path, threshold_value):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Resize the image while maintaining its aspect ratio
#     resized_image = cv2.resize(image, (desired_width, desired_height))

#     # Convert the resized image to grayscale
#     gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

#     # Manual thresholding with a fixed threshold_value
#     _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

#     # Invert the binary mask
#     inverted_mask = cv2.bitwise_not(binary_mask)

#     # Create a 3-channel color mask
#     color_mask = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

#     # Use the color mask to extract the object
#     object_retained_color = cv2.bitwise_and(resized_image, color_mask)

#     return object_retained_color

# # Get a list of all JPG files in the source folder
# jpg_files = glob.glob(os.path.join(source_folder, '*.jpg'))

# # Loop through each JPG file and perform background removal
# for jpg_file in jpg_files:
#     # Define a threshold value for each image (you can customize this per image)
#     threshold_value = 175  # Adjust this value based on your images

#     # Perform background removal
#     result = remove_background(jpg_file, threshold_value)

#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # Define the output path for the result
#     output_path = os.path.join(destination_folder, os.path.basename(jpg_file))

#     # Save the result to the destination folder
#     cv2.imwrite(output_path, result)

# print("Background removal and saving complete.")


# ############# To remove the gray part for x40 (working best) #########
# from PIL import Image, ImageStat
# import os

# # Function to calculate the average pixel color of an image region
# def average_color(image, left, upper, right, lower):
#     region = image.crop((left, upper, right, lower))
#     stat = ImageStat.Stat(region)
#     return stat.mean[:3]

# # Function to check if an image region has a uniform color
# def is_uniform_color(image, left, upper, right, lower, target_color, tolerance=10):
#     color = average_color(image, left, upper, right, lower)
#     return all(abs(color[i] - target_color[i]) <= tolerance for i in range(3))

# # Function to check if the central region of an image has a balanced color
# def is_balanced_color(image, target_color, tolerance=10):
#     width, height = image.size
#     center_left = width // 4
#     center_upper = height // 4
#     center_right = 3 * (width // 4)
#     center_lower = 3 * (height // 4)
#     center_color = average_color(image, center_left, center_upper, center_right, center_lower)
#     # Check if the central region color is different from the target color
#     return not all(abs(center_color[i] - target_color[i]) <= tolerance for i in range(3))

# # Function to crop and save regions from an image
# def crop_and_save_all_coordinates(image_path, output_folder, crop_size, target_color, tolerance=10):
#     # Open the image
#     img = Image.open(image_path)

#     # Get the size of the image
#     width, height = img.size

#     # Get the base filename without extension
#     base_filename = os.path.splitext(os.path.basename(image_path))[0]

#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Convert the target hex color to RGB
#     target_color = tuple(int(target_color[i:i+2], 16) for i in (1, 3, 5))

#     # Loop through the image and crop 250x250 regions for every coordinate
#     for y in range(0, height - crop_size + 1, crop_size):
#         for x in range(0, width - crop_size + 1, crop_size):
#             left = x
#             upper = y
#             right = x + crop_size
#             lower = y + crop_size

#             # Check if the region has a uniform color that matches the target color
#             if not is_uniform_color(img, left, upper, right, lower, target_color, tolerance):
#                 # Crop the non-uniform color region
#                 cropped_region = img.crop((left, upper, right, lower))

#                 # Check if the central region has a balanced color
#                 if is_balanced_color(cropped_region, target_color, tolerance):
#                     # Generate a unique filename for the cropped region
#                     output_filename = os.path.join(output_folder, f"{base_filename}_crop_{x}_{y}.jpg")

#                     # Save the cropped region
#                     cropped_region.save(output_filename)
#                     print(f"Saved {output_filename}")

# # Function to crop a list of images in a folder
# def crop_images_in_folder(input_folder, output_folder, crop_size, target_color, tolerance=10):
#     # Get a list of all files in the input folder
#     files = os.listdir(input_folder)

#     # Iterate through the files and crop each image
#     for file in files:
#         if file.endswith(('.JPG', '.jpeg', '.png', '.bmp', '.gif')):
#             input_image_path = os.path.join(input_folder, file)
#             crop_and_save_all_coordinates(input_image_path, output_folder, crop_size, target_color, tolerance)

# if __name__ == "__main__":
#     # Specify the input folder containing images, the output folder for cropped images,
#     # and the fixed crop size (250x250 pixels).
#     input_folder = "E:\CodingProjects\machine_learning\Experiment 2\Zarrouk\Day_1_Zarrouk\Batch_1"      # Change this to your input folder path
#     output_folder = "E:\CodingProjects\machine_learning\Experiment 2\Zarrouk\Day_1_Zarrouk\Batch_1_Crop_X100"    # Change this to your desired output folder
#     crop_size = 250

#     # Specify the target color in hex format (e.g., #C1C0C5)
#     target_color_hex = "#C1C0C5"

#     # Call the crop_images_in_folder function
#     crop_images_in_folder(input_folder, output_folder, crop_size, target_color_hex)

############# To remove the gray part for x100 (working best) ########################
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













