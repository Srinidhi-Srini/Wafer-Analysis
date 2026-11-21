import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import RBFInterpolator

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_DIR = "input_images"
IMG_OUT_DIR = "generated_images"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

IMAGE_PATH = os.path.join(INPUT_DIR, "wafer_image3.png")
WAFER_DIAMETER_MM = 100.0   # Change to your wafer size (e.g., 50.8, 76.2, 100.0, 150.0 mm)
DROPLET_VOLUME_UL = 20.0    # 20 uL = 20 mm^3
WATER_GAMMA_L = 72.8        # mN/m at 20 C

# ==========================================
# 2. PHYSICS EQUATIONS
# ==========================================
def solve_contact_angle_deg(a_mm, vol_mm3=DROPLET_VOLUME_UL):
    """
    Solves the exact spherical cap volume equation for theta (in degrees):
    V = (pi * a^3 / 3) * ((2 + cos(theta)) * (1 - cos(theta))^2) / sin^3(theta)
    """
    def residual(theta_rad):
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        v_calc = (np.pi * (a_mm ** 3) / 3.0) * ((2.0 + c) * ((1.0 - c) ** 2)) / (s ** 3)
        return v_calc - vol_mm3

    # Search for theta between 0.5 deg and 175 deg
    try:
        theta_rad = brentq(residual, np.radians(0.5), np.radians(175.0))
        return np.degrees(theta_rad)
    except ValueError:
        return np.nan  # Droplet radius out of physical bounds for given volume

def owrk_surface_energy(theta_water_deg, theta_diiodo_deg):
    """
    Ready for when you add diiodomethane measurements.
    Returns (gamma_s_dispersive, gamma_s_polar, gamma_s_total) in mN/m.
    """
    cos_w = np.cos(np.radians(theta_water_deg))
    cos_d = np.cos(np.radians(theta_diiodo_deg))

    gamma_sd = (50.8 * (1 + cos_d) ** 2) / 4.0
    gamma_sp = ((72.8 * (1 + cos_w) - 2 * np.sqrt(21.8 * gamma_sd)) / (2 * np.sqrt(51.0))) ** 2
    return gamma_sd, gamma_sp, gamma_sd + gamma_sp

# ==========================================
# 3. IMAGE PROCESSING & DROPLET DETECTION
# ==========================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not find '{IMAGE_PATH}'.")

h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

H, S, V = cv2.split(hsv)
L, A, B = cv2.split(lab)

# --- Step A: Detect Wafer Circle via Purple Color Mask ---
# The wafer has a distinct magenta/purple tint (high A channel in Lab)
wafer_color_mask = ((A > 136) & (V > 50)).astype(np.uint8) * 255
kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
wafer_color_mask = cv2.morphologyEx(wafer_color_mask, cv2.MORPH_CLOSE, kernel_big)

wafer_contours, _ = cv2.findContours(wafer_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not wafer_contours:
    raise RuntimeError("Wafer not detected. Check the A > 136 threshold.")

# Combine all wafer strip contours into one convex hull to get the true outer circle
all_wafer_pts = np.vstack([c for c in wafer_contours if cv2.contourArea(c) > 500])
hull = cv2.convexHull(all_wafer_pts)
(wafer_cx, wafer_cy), wafer_r_px = cv2.minEnclosingCircle(hull)
wafer_cx, wafer_cy, wafer_r_px = int(round(wafer_cx)), int(round(wafer_cy)), int(round(wafer_r_px))

mm_per_px = (WAFER_DIAMETER_MM / 2.0) / wafer_r_px

# Slightly shrink wafer mask to ignore the outer rim
wafer_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(wafer_mask, (wafer_cx, wafer_cy), int(wafer_r_px * 0.97), 255, -1)

# --- Step B: Segment Silvery Droplets using Lab + HSV ---
# Droplets are:
#   - Inside the wafer
#   - Neutral/silvery (low saturation S, low magenta A)
#   - Not black seams (V > 120) and not blown-out white glare (V < 245)
droplet_mask = (
    (wafer_mask > 0) &
    (S < 35) &
    (A < 134) &
    (V > 120) &
    (V < 245)
).astype(np.uint8) * 255

# Clean up noise and separate any droplets barely touching glare/seams
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
droplet_mask = cv2.morphologyEx(droplet_mask, cv2.MORPH_OPEN, kernel_open)
droplet_mask = cv2.morphologyEx(droplet_mask, cv2.MORPH_CLOSE, kernel_close)

# Save mask for easy debugging
cv2.imwrite(os.path.join(IMG_OUT_DIR, "debug_mask.jpg"), droplet_mask)

# ==========================================
# 4. EXTRACT DATA & CALCULATE METRICS
# ==========================================
contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_r_px = max(5, int(2.0 / mm_per_px))
max_r_px = int(6.5 / mm_per_px)
min_area = np.pi * (min_r_px ** 2)
max_area = np.pi * (max_r_px ** 2)

records = []
debug_img = img.copy()
cv2.circle(debug_img, (wafer_cx, wafer_cy), wafer_r_px, (0, 255, 0), 2)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if not (min_area <= area <= max_area):
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    # Filter out elongated glare strips using circularity and solidity
    circularity = 4 * np.pi * area / (perimeter ** 2)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0

    if circularity < 0.60 or solidity < 0.85:
        continue

    # Centroid and equivalent circular radius
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx_px = int(M["m10"] / M["m00"])
    cy_px = int(M["m01"] / M["m00"])
    r_px = np.sqrt(area / np.pi)

    x_mm = (cx_px - wafer_cx) * mm_per_px
    y_mm = -(cy_px - wafer_cy) * mm_per_px
    a_mm = r_px * mm_per_px

    theta_deg = solve_contact_angle_deg(a_mm, DROPLET_VOLUME_UL)
    if np.isnan(theta_deg):
        continue

    work_of_adhesion = WATER_GAMMA_L * (1 + np.cos(np.radians(theta_deg)))

    records.append({
        "droplet_id": len(records) + 1,
        "x_mm": round(x_mm, 2),
        "y_mm": round(y_mm, 2),
        "base_radius_mm": round(a_mm, 3),
        "contact_angle_deg": round(theta_deg, 2),
        "work_of_adhesion_mN_m": round(work_of_adhesion, 2)
    })

    cv2.circle(debug_img, (cx_px, cy_px), int(round(r_px)), (0, 0, 255), 2)
    cv2.putText(debug_img, f"{theta_deg:.1f}", (cx_px - 15, cy_px),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("0 droplets detected. Inspect generated_images/debug_mask.jpg to tune S, A, and V thresholds.")
# ==========================================
# 5. PLOT X-Y WAFER MAP
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6))

wafer_circle = plt.Circle((0, 0), WAFER_DIAMETER_MM / 2.0, color='gray', fill=False, lw=2, ls='--')
ax.add_patch(wafer_circle)

sc = ax.scatter(
    df["x_mm"], df["y_mm"],
    c=df["contact_angle_deg"], cmap="viridis_r",
    vmin=10, vmax=35,
    s=df["base_radius_mm"] * 40, edgecolors="black", alpha=0.9
)

plt.colorbar(sc, ax=ax, extend="max", label="Water Contact Angle (°)")
ax.set_xlim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax.set_ylim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax.set_aspect("equal")
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_title("Silicon Wafer Contact Angle Map (20 µL Water)")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(IMG_OUT_DIR, "wafer_map.png"), dpi=300)

# ==========================================
# 6. PLOT SMOOTH GRADIENT HEATMAP (NO DOTS)
# ==========================================
R = WAFER_DIAMETER_MM / 2.0

grid_res = 300
xi = np.linspace(-R, R, grid_res)
yi = np.linspace(-R, R, grid_res)
X, Y = np.meshgrid(xi, yi)

points = df[["x_mm", "y_mm"]].values
values = df["contact_angle_deg"].values
rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=2.0)

grid_points = np.column_stack([X.ravel(), Y.ravel()])
Z = rbf(grid_points).reshape(X.shape)
Z[X**2 + Y**2 > R**2] = np.nan

fig2, ax2 = plt.subplots(figsize=(7, 6))

im = ax2.imshow(
    Z, extent=[-R, R, -R, R], origin="lower",
    cmap="viridis_r", vmin=10, vmax=35
)

wafer_outline = plt.Circle((0, 0), R, color="black", fill=False, lw=2)
ax2.add_patch(wafer_outline)

plt.colorbar(im, ax=ax2, extend="max", label="Water Contact Angle (°)")
ax2.set_xlim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax2.set_ylim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax2.set_aspect("equal")
ax2.set_xlabel("X Position (mm)")
ax2.set_ylabel("Y Position (mm)")
ax2.set_title("Silicon Wafer Contact Angle Gradient (20 µL Water)")
ax2.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(IMG_OUT_DIR, "wafer_gradient_map.png"), dpi=300)

plt.show()