import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import RBFInterpolator

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMAGE_PATH = "wafer_image.png"
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
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# --- Step A: Detect Wafer Circle ---
# Blur and use HoughCircles (you may need to tune param1, param2, minRadius, maxRadius)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 2,
    param1=50, param2=30, minRadius=int(min(h, w) * 0.15), maxRadius=int(min(h, w) * 0.45)
)

if circles is None:
    raise RuntimeError("Wafer circle not detected. Adjust HoughCircles parameters or set center/radius manually.")

wafer_cx, wafer_cy, wafer_r_px = np.round(circles[0, 0]).astype(int)
mm_per_px = (WAFER_DIAMETER_MM / 2.0) / wafer_r_px

# Create a circular mask slightly inside the wafer edge to ignore edge glare
wafer_mask = np.zeros_like(gray)
# 1. In Step A: Expand the mask from 0.96 to 0.99 so outer edge columns aren't clipped
cv2.circle(wafer_mask, (wafer_cx, wafer_cy), int(wafer_r_px * 0.99), 255, -1)

# --- Step B: Enhance Contrast & Detect Circular Droplets ---
# CLAHE boosts local droplet edges in both dark purple and bright glare zones
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
blurred_drops = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

# Expected radius range for a 20 uL water droplet (roughly 2.0 mm to 6.5 mm)
min_r_px = max(5, int(2.0 / mm_per_px))
max_r_px = int(6.5 / mm_per_px)
min_dist_px = int(7.5 / mm_per_px)

drop_circles = cv2.HoughCircles(
    blurred_drops,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=min_dist_px,
    param1=50,
    param2=26,   # Was 22
    minRadius=min_r_px,
    maxRadius=max_r_px
)

# ==========================================
# 4. EXTRACT DATA & CALCULATE METRICS
# ==========================================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
records = []
accepted_circles = []
debug_img = img.copy()
cv2.circle(debug_img, (wafer_cx, wafer_cy), wafer_r_px, (0, 255, 0), 2)

if drop_circles is not None:
    # HoughCircles returns the strongest (most likely real) circles first
    for (cx_px, cy_px, r_px) in np.round(drop_circles[0, :]).astype(int):
        # 1. Must be inside wafer mask
        if not (0 <= cy_px < h and 0 <= cx_px < w and wafer_mask[cy_px, cx_px] > 0):
            continue

        # 2. Reject circles that overlap an already-accepted stronger circle
        if any(np.hypot(cx_px - ax, cy_px - ay) < 0.88 * (r_px + ar)
               for (ax, ay, ar) in accepted_circles):
            continue

        # 3. Check interior color (real droplets are neutral gray/white, not purple or black)
        half = max(2, int(r_px * 0.45))
        roi = hsv[max(0, cy_px - half):min(h, cy_px + half),
                  max(0, cx_px - half):min(w, cx_px + half)]
        median_s = np.median(roi[:, :, 1])
        median_v = np.median(roi[:, :, 2])

        # Reject purple wafer background (high saturation) and dark seams (low brightness)
        if median_s > 45 or median_v < 85:
            continue

        accepted_circles.append((cx_px, cy_px, r_px))

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

        cv2.circle(debug_img, (cx_px, cy_px), r_px, (0, 0, 255), 2)
        cv2.putText(debug_img, f"{theta_deg:.1f}", (cx_px - 15, cy_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("0 droplets detected. Try raising median_s (e.g., to 80) or lowering param2.")

# Auto-center so the droplet closest to the middle lands at (0, 0)
center_idx = (df["x_mm"]**2 + df["y_mm"]**2).idxmin()
x_offset = df.loc[center_idx, "x_mm"]
y_offset = df.loc[center_idx, "y_mm"]

df["x_mm"] = (df["x_mm"] - x_offset).round(2)
df["y_mm"] = (df["y_mm"] - y_offset).round(2)

df.to_csv("wafer_droplets.csv", index=False)
cv2.imwrite("debug_detection.jpg", debug_img)
print(f"Shifted by ({-x_offset:.2f}, {-y_offset:.2f}) mm. Detected {len(df)} droplets.")

# ==========================================
# 5. PLOT X-Y WAFER MAP
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6))

# Draw wafer outline
wafer_circle = plt.Circle((0, 0), WAFER_DIAMETER_MM / 2.0, color='gray', fill=False, lw=2, ls='--')
ax.add_patch(wafer_circle)

# Scatter plot of droplets colored by Contact Angle (or Work of Adhesion)
sc = ax.scatter(
    df["x_mm"], df["y_mm"],
    c=df["contact_angle_deg"], cmap="viridis_r",
    vmin=10, vmax=35,  # Locks the color scale to your real droplet range
    s=df["base_radius_mm"] * 40, edgecolors="black", alpha=0.9
)

plt.colorbar(sc, ax=ax, label="Water Contact Angle (°)")
ax.set_xlim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax.set_ylim(-WAFER_DIAMETER_MM * 0.55, WAFER_DIAMETER_MM * 0.55)
ax.set_aspect("equal")
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_title("Silicon Wafer Contact Angle Map (20 µL Water)")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("wafer_map.png", dpi=300)

# ==========================================
# 6. PLOT SMOOTH GRADIENT HEATMAP (NO DOTS)
# ==========================================
R = WAFER_DIAMETER_MM / 2.0

# Create a fine 2D grid covering the wafer
grid_res = 300
xi = np.linspace(-R, R, grid_res)
yi = np.linspace(-R, R, grid_res)
X, Y = np.meshgrid(xi, yi)

# Interpolate contact angles smoothly across the entire wafer
points = df[["x_mm", "y_mm"]].values
values = df["contact_angle_deg"].values
rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=2.0)

grid_points = np.column_stack([X.ravel(), Y.ravel()])
Z = rbf(grid_points).reshape(X.shape)

# Mask out everything outside the circular wafer boundary
Z[X**2 + Y**2 > R**2] = np.nan

fig2, ax2 = plt.subplots(figsize=(7, 6))

# Plot smooth gradient
im = ax2.imshow(
    Z, extent=[-R, R, -R, R], origin="lower",
    cmap="viridis_r", vmin=10, vmax=35
)

# Draw wafer outline
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
plt.savefig("wafer_gradient_map.png", dpi=300)

plt.show()