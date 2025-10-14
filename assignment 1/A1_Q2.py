# === A1_Q2.py : Invisible Cloak (Red Version) ===
# This program detects a red cloak and replaces it with background pixels
# using OpenCV’s HSV color segmentation (as taught in class).

import cv2
import numpy as np

# === Step 1: Convert frame to HSV and create cloak mask ===
def apply_invisible_cloak(frame, background):
    # Convert the frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red cloak
    # (Red wraps around hue=0°, so two ranges are needed)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Clean the mask (remove small noise)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert mask to get non-red regions
    mask_inv = cv2.bitwise_not(mask)

    # Extract background where cloak is detected
    cloak_area = cv2.bitwise_and(background, background, mask=mask)

    # Extract visible (non-red) parts of the frame
    visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both to form the final frame
    final = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

    return final


# === Step 2: Main program ===
def main():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("No webcam detected — exiting.")
        return

    print("Press 'b' to capture background, then show red cloth.")
    background = None

    # Capture background (scene without cloak)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Background Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            background = frame.copy()
            print("Background captured.")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Start invisible cloak effect
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = apply_invisible_cloak(frame, background)
        cv2.imshow("Cloak Effect", output)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
