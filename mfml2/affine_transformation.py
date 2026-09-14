import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# FUNCTION TO DISPLAY TRANSFORMATION MATRIX
# =========================================================

def print_matrix(matrix, title):

    print("\n" + title + ":")

    for row in matrix:
        for value in row:
            print(f"{value:8.2f}", end=" ")
        print()


# =========================================================
# TRANSLATION MATRIX
# =========================================================

def translation_matrix(tx, ty):

    matrix = np.array([
        [1,  0,  tx],
        [0,  1,  ty],
        [0,  0,   1]
    ])

    return matrix


# =========================================================
# ROTATION MATRIX
# =========================================================

def rotation_matrix(angle):

    # Convert degrees to radians
    radians = math.radians(angle)

    cos_value = math.cos(radians)
    sin_value = math.sin(radians)

    matrix = np.array([
        [cos_value, -sin_value, 0],
        [sin_value,  cos_value, 0],
        [0,          0,         1]
    ])

    return matrix


# =========================================================
# SCALING MATRIX
# =========================================================

def scaling_matrix(sx, sy):

    matrix = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])

    return matrix


# =========================================================
# SHEARING MATRIX
# =========================================================

def shearing_matrix(shx, shy):

    matrix = np.array([
        [1,  shx, 0],
        [shy, 1,  0],
        [0,   0,  1]
    ])

    return matrix


# =========================================================
# APPLY AFFINE TRANSFORMATION
# =========================================================

def apply_transformation(image, matrix):

    # Convert image to NumPy array
    image_array = np.array(image)

    height = image_array.shape[0]
    width = image_array.shape[1]

    # Create empty output image
    output = np.zeros_like(image_array)

    # Find center of image
    center_x = width / 2
    center_y = height / 2

    # Matrix to move center to origin
    move_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])

    # Matrix to move origin back to center
    move_back = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])

    # Create transformation around image center
    final_matrix = np.dot(
        move_back,
        np.dot(matrix, move_to_origin)
    )

    # Find inverse matrix
    inverse_matrix = np.linalg.inv(final_matrix)

    # Go through every pixel
    for y in range(height):

        for x in range(width):

            # Homogeneous coordinate
            point = np.array([
                x,
                y,
                1
            ])

            # Find corresponding original point
            original_point = np.dot(
                inverse_matrix,
                point
            )

            original_x = int(round(original_point[0]))
            original_y = int(round(original_point[1]))

            # Check if point is inside image
            if (
                original_x >= 0
                and original_x < width
                and original_y >= 0
                and original_y < height
            ):

                output[y][x] = image_array[
                    original_y
                ][original_x]

    return Image.fromarray(output)


# =========================================================
# LOAD IMAGE
# =========================================================

image_path = "input.jpg"

try:

    image = Image.open(image_path)

    # Convert to RGB
    image = image.convert("RGB")

except FileNotFoundError:

    print("Error: Image file not found.")
    print("Make sure input.jpg is in the same folder.")
    exit()


print("=" * 60)
print("DATA AUGMENTATION USING AFFINE TRANSFORMATION")
print("=" * 60)


# =========================================================
# USER PARAMETERS
# =========================================================

tx = 50
ty = 30

angle = 30

sx = 1.5
sy = 1.5

shx = 0.3
shy = 0.0


# =========================================================
# CREATE TRANSFORMATION MATRICES
# =========================================================

translation = translation_matrix(tx, ty)

rotation = rotation_matrix(angle)

scaling = scaling_matrix(sx, sy)

shearing = shearing_matrix(shx, shy)


# =========================================================
# DISPLAY MATRICES
# =========================================================

print_matrix(
    translation,
    "Translation Matrix"
)

print_matrix(
    rotation,
    "Rotation Matrix"
)

print_matrix(
    scaling,
    "Scaling Matrix"
)

print_matrix(
    shearing,
    "Shearing Matrix"
)


# =========================================================
# APPLY TRANSFORMATIONS
# =========================================================

translated_image = apply_transformation(
    image,
    translation
)

rotated_image = apply_transformation(
    image,
    rotation
)

scaled_image = apply_transformation(
    image,
    scaling
)

sheared_image = apply_transformation(
    image,
    shearing
)


# =========================================================
# COMBINE TRANSFORMATIONS
# =========================================================

combined_matrix = np.dot(
    translation,
    rotation
)

combined_matrix = np.dot(
    combined_matrix,
    scaling
)


print_matrix(
    combined_matrix,
    "Combined Transformation Matrix"
)


combined_image = apply_transformation(
    image,
    combined_matrix
)


# =========================================================
# SAVE AUGMENTED IMAGES
# =========================================================

translated_image.save("translated.jpg")

rotated_image.save("rotated.jpg")

scaled_image.save("scaled.jpg")

sheared_image.save("sheared.jpg")

combined_image.save("combined.jpg")


print("\nAugmented images saved successfully.")


# =========================================================
# DISPLAY IMAGES
# =========================================================

plt.figure(figsize=(12, 8))


# Original image
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")


# Translated image
plt.subplot(2, 3, 2)
plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis("off")


# Rotated image
plt.subplot(2, 3, 3)
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis("off")


# Scaled image
plt.subplot(2, 3, 4)
plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis("off")


# Sheared image
plt.subplot(2, 3, 5)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis("off")


# Combined image
plt.subplot(2, 3, 6)
plt.imshow(combined_image)
plt.title("Combined Transformation")
plt.axis("off")


plt.tight_layout()

plt.show()