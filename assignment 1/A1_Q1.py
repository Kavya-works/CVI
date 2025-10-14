import cv2
import numpy as np
import math
import argparse

# === 1. Create white canvas ===
# Create a 600x600 white background image
img = np.ones((600, 600, 3), dtype=np.uint8) * 255

# === 2. Define colours in BGR format ===
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE  = (255, 0, 0)
BLACK = (0, 0, 0)

# === 3. Setup positions for the three coloured arcs ===
canvas_w, canvas_h = 600, 600
cx = canvas_w // 2

# The main triangle area where arcs will be arranged
triangle_cx, triangle_cy = cx, 210

# Circle size and thickness for each arc
radius = 60
thickness = 28

# Compute positions for the 3 circle centres (120° apart)
# Red (top), Blue (bottom-right), Green (bottom-left)
circum_r = 95
angles_deg = [-90, 30, 150]
centers = []
for a in angles_deg:
    rad = math.radians(a)
    cx_i = int(triangle_cx + circum_r * math.cos(rad))
    cy_i = int(triangle_cy + circum_r * math.sin(rad))
    centers.append((cx_i, cy_i))

center_red = centers[0]
center_blue = centers[1]
center_green = centers[2]

# === 4. Draw coloured arcs for the OpenCV swirl ===
# Each arc has an open gap facing toward the centre of the triangle
circle_info = [
    (center_red, RED),
    (center_green, GREEN),
    (center_blue, BLUE),
]

gap_half = 50  # size of the white gap between arcs
gap_points = []

for (cx_i, cy_i), color in circle_info:
    # Angle from circle centre toward the middle of the triangle
    phi = math.degrees(math.atan2(triangle_cy - cy_i, triangle_cx - cx_i))

    # Define the open (white) area and draw the coloured arc opposite to it
    gap_start = phi - gap_half
    gap_end = phi + gap_half
    arc_start = gap_end
    arc_end = gap_start + 360
    cv2.ellipse(
        img, (cx_i, cy_i), (radius, radius), 0,
        float(arc_start), float(arc_end), color, thickness, lineType=cv2.LINE_AA
    )

    # Record inner points for the white triangular gap
    rad_phi = math.radians(phi)
    px = int(cx_i + (radius - thickness // 2) * math.cos(rad_phi))
    py = int(cy_i + (radius - thickness // 2) * math.sin(rad_phi))
    gap_points.append((px, py))

# === 5. Create the central white triangle (gap between arcs) ===
tri_pts = np.array(gap_points, dtype=np.int32)
cv2.fillConvexPoly(img, tri_pts, (255, 255, 255), lineType=cv2.LINE_AA)

# Add rounded white notches for a smoother inner edge
notch_r = thickness // 2 + 3
for (px, py) in gap_points:
    cv2.circle(img, (px, py), notch_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)

# === 6. Add the text "OpenCV" below the logo ===
text = "OpenCV"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1.4
t = 3
text_size, _ = cv2.getTextSize(text, font, scale, t)
text_x = (canvas_w - text_size[0]) // 2
text_y = 400
cv2.putText(img, text, (text_x, text_y), font, scale, BLACK, t, cv2.LINE_AA)

# === 7. Display and save the image ===
def main(no_window: bool = False):
    """Display and save the generated OpenCV logo image."""
    out_name = "A1_Q1_final_logo.png"
    cv2.imwrite(out_name, img)
    if not no_window:
        cv2.imshow("OpenCV Logo", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# === 8. Run main program ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw OpenCV logo using OpenCV drawing functions")
    parser.add_argument('--no-window', action='store_true', help='Only save the image without opening a window')
    args = parser.parse_args()
    main(no_window=args.no_window)
