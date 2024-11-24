import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Preprocess the image: grayscale, threshold, and noise removal."""
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove small noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return image, gray, clean

def segment_lines(binary_image):
    """Segment lines of text using horizontal projection."""
    horizontal_sum = np.sum(binary_image, axis=1)
    lines = []
    in_line = False
    for i, value in enumerate(horizontal_sum):
        if value > 0 and not in_line:
            start = i
            in_line = True
        elif value == 0 and in_line:
            end = i
            lines.append((start, end))
            in_line = False
    return lines

def segment_words(line_image):
    """Segment words within a line using vertical projection."""
    vertical_sum = np.sum(line_image, axis=0)
    words = []
    in_word = False
    for i, value in enumerate(vertical_sum):
        if value > 0 and not in_word:
            start = i
            in_word = True
        elif value == 0 and in_word:
            end = i
            words.append((start, end))
            in_word = False
    return words

def save_segments(line_image, words, output_folder, line_index):
    """Save segmented word images to the output folder."""
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    
    for idx, (start, end) in enumerate(words):
        word_image = line_image[:, start:end]
        output_path = os.path.join(output_folder, f"line{line_index + 1}_word{idx + 1}.png")
        cv2.imwrite(output_path, word_image)
        print(f"Saved word {idx + 1} of line {line_index + 1} at: {output_path}")

# Main script
image_path = "Hand.jpg"  # Replace with your image path
output_folder = "segmented_words"  # Folder to save word images

# Preprocess the image
original, gray, binary = preprocess_image(image_path)

# Line segmentation
lines = segment_lines(binary)
print(f"Detected {len(lines)} lines of text.")

# Word segmentation and saving words
for line_index, (line_start, line_end) in enumerate(lines):
    line_image = binary[line_start:line_end, :]  # Extract line image
    words = segment_words(line_image)
    print(f"Detected {len(words)} words in line {line_index + 1}.")
    
    # Save word images
    save_segments(line_image, words, output_folder, line_index)
