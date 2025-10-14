# === A1_PhotoEditor.py : Photo Booth Application ===
# Modular photo editor using OpenCV + NumPy
# Functions: brightness, contrast, grayscale, padding, thresholding, undo, save

import cv2
import numpy as np
import matplotlib.pyplot as plt


# === Operation Functions ===
def adjust_brightness(img, value):
    """Adjust brightness: positive = brighter, negative = darker"""
    return cv2.convertScaleAbs(img, beta=value)


def adjust_contrast(img, factor):
    """Adjust contrast: >1 increases contrast, <1 decreases"""
    return cv2.convertScaleAbs(img, alpha=factor)


def convert_to_grayscale(img):
    """Convert to grayscale but keep 3-channel image for uniformity"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_padding(img, size, border_type):
    """Add border padding to image"""
    types = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE
    }
    return cv2.copyMakeBorder(img, size, size, size, size,
                              types.get(border_type, cv2.BORDER_CONSTANT))


def apply_threshold(img, thresh_type, value):
    """Apply binary or inverse threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY if thresh_type == "binary" else cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(gray, value, 255, mode)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# === Helper Display Function ===
def show_side_by_side(original, preview, title1="Original", title2="Preview"):
    """Display before/after side by side using matplotlib"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis('off')

    plt.show()


# === Main App ===
def main():
    # Step 1: Load image
    img_path = input("Enter image filename: ").strip()
    image = cv2.imread(img_path)

    if image is None:
        print("Could not load image. Make sure the file is in this folder.")
        return

    history = [image.copy()]  # keep copies for undo

    # Step 2: Interactive menu
    while True:
        print("""
==== Mini Photo Editor ====
1. Adjust Brightness
2. Adjust Contrast
3. Convert to Grayscale
4. Add Padding
5. Apply Thresholding
6. Undo Last Operation
7. Save Image and Exit
        """)

        choice = input("Choose an option (1–7): ").strip()

        if choice == "1":
            val = int(input("Enter brightness adjustment (-100 to 100): "))
            new_img = adjust_brightness(image, val)

        elif choice == "2":
            factor = float(input("Enter contrast factor (0.0 to 3.0): "))
            new_img = adjust_contrast(image, factor)

        elif choice == "3":
            new_img = convert_to_grayscale(image)

        elif choice == "4":
            size = int(input("Enter padding size (px): "))
            btype = input("Border type (constant/reflect/replicate): ").strip().lower()
            new_img = add_padding(image, size, btype)

        elif choice == "5":
            mode = input("Threshold type (binary/inverse): ").strip().lower()
            value = int(input("Enter threshold value (0–255): "))
            new_img = apply_threshold(image, mode, value)

        elif choice == "6":
            if len(history) > 1:
                history.pop()
                image = history[-1].copy()
                print("Undone last operation.")
                continue
            else:
                print("Nothing to undo.")
                continue

        elif choice == "7":
            filename = input("Enter filename to save (e.g., result.jpg): ")
            cv2.imwrite(filename, image)
            print("Image saved successfully as", filename)
            break

        else:
            print("Invalid choice. Try again.")
            continue

        # Display result
        show_side_by_side(image, new_img)
        print("Press any key inside the preview window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image = new_img
        history.append(image.copy())


# === Run the program ===
if __name__ == "__main__":
    main()
