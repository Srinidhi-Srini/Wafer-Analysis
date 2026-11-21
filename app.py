import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import brentq
from scipy.interpolate import RBFInterpolator

st.set_page_config(page_title="Wafer Contact Angle Mapper", layout="wide")
st.title("Silicon Wafer Contact Angle Mapper")

# ==========================================
# 1. PHYSICS EQUATIONS
# ==========================================
def solve_contact_angle_deg(a_mm, vol_mm3):
    def residual(theta_rad):
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        v_calc = (np.pi * (a_mm ** 3) / 3.0) * ((2.0 + c) * ((1.0 - c) ** 2)) / (s ** 3)
        return v_calc - vol_mm3

    try:
        theta_rad = brentq(residual, np.radians(0.5), np.radians(175.0))
        return np.degrees(theta_rad)
    except ValueError:
        return np.nan

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Wafer & Droplet Settings")
    wafer_diameter_mm = st.number_input("Wafer Diameter (mm)", value=100.0, step=10.0)
    droplet_volume_ul = st.number_input("Droplet Volume (µL)", value=20.0, step=1.0)
    water_gamma_l = 72.8
    auto_center = st.checkbox("Auto-center on middle droplet", value=True)

    st.header("Detection Thresholds")
    wafer_a_min = st.slider("Wafer Purple (Lab A >)", 125, 160, 136)
    drop_s_max = st.slider("Droplet Max Saturation (S <)", 10, 100, 35)
    drop_a_max = st.slider("Droplet Max Magenta (A <)", 125, 150, 134)
    drop_v_min = st.slider("Droplet Min Brightness (V >)", 50, 200, 120)
    drop_v_max = st.slider("Droplet Max Brightness (V <)", 200, 255, 245)

    st.header("Shape & Plot Settings")
    min_circ = st.slider("Min Circularity", 0.30, 0.95, 0.60, 0.05)
    min_sol = st.slider("Min Solidity", 0.50, 0.99, 0.85, 0.05)
    vmin, vmax = st.slider("Colorbar Range (°)", 0, 90, (10, 35))

# ==========================================
# 3. FILE UPLOAD & PROCESSING
# ==========================================
uploaded_file = st.file_uploader("Upload a wafer image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("Upload a wafer image above to generate the contact angle maps.")
    st.stop()

# Decode uploaded image directly into OpenCV format
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
H, S, V = cv2.split(hsv)
L, A, B = cv2.split(lab)

# --- Step A: Detect Wafer Circle ---
wafer_color_mask = ((A > wafer_a_min) & (V > 50)).astype(np.uint8) * 255
kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
wafer_color_mask = cv2.morphologyEx(wafer_color_mask, cv2.MORPH_CLOSE, kernel_big)

wafer_contours, _ = cv2.findContours(wafer_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_wafer_cnts = [c for c in wafer_contours if cv2.contourArea(c) > 500]

if not valid_wafer_cnts:
    st.error("Wafer not detected. Try lowering the 'Wafer Purple (Lab A >)' slider in the sidebar.")
    st.stop()

all_wafer_pts = np.vstack(valid_wafer_cnts)
hull = cv2.convexHull(all_wafer_pts)
(wafer_cx, wafer_cy), wafer_r_px = cv2.minEnclosingCircle(hull)
wafer_cx, wafer_cy, wafer_r_px = int(round(wafer_cx)), int(round(wafer_cy)), int(round(wafer_r_px))
mm_per_px = (wafer_diameter_mm / 2.0) / wafer_r_px

wafer_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(wafer_mask, (wafer_cx, wafer_cy), int(wafer_r_px * 0.97), 255, -1)

# --- Step B: Segment Droplets ---
droplet_mask = (
    (wafer_mask > 0) &
    (S < drop_s_max) &
    (A < drop_a_max) &
    (V > drop_v_min) &
    (V < drop_v_max)
).astype(np.uint8) * 255

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
droplet_mask = cv2.morphologyEx(droplet_mask, cv2.MORPH_OPEN, kernel_open)
droplet_mask = cv2.morphologyEx(droplet_mask, cv2.MORPH_CLOSE, kernel_close)

# --- Step C: Extract Data ---
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

    circularity = 4 * np.pi * area / (perimeter ** 2)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0

    if circularity < min_circ or solidity < min_sol:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx_px = int(M["m10"] / M["m00"])
    cy_px = int(M["m01"] / M["m00"])
    r_px = np.sqrt(area / np.pi)

    x_mm = (cx_px - wafer_cx) * mm_per_px
    y_mm = -(cy_px - wafer_cy) * mm_per_px
    a_mm = r_px * mm_per_px

    theta_deg = solve_contact_angle_deg(a_mm, droplet_volume_ul)
    if np.isnan(theta_deg):
        continue

    work_of_adhesion = water_gamma_l * (1 + np.cos(np.radians(theta_deg)))

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

# ==========================================
# 4. DISPLAY DEBUG IMAGES
# ==========================================
st.subheader(f"Detection Preview ({len(df)} droplets detected)")
col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Detected Droplets", use_container_width=True)
with col2:
    st.image(droplet_mask, caption="Droplet Segmentation Mask", use_container_width=True)

if df.empty:
    st.warning("0 droplets detected. Adjust the S, A, and V sliders in the sidebar while watching the mask.")
    st.stop()

if auto_center:
    center_idx = (df["x_mm"] ** 2 + df["y_mm"] ** 2).idxmin()
    df["x_mm"] = (df["x_mm"] - df.loc[center_idx, "x_mm"]).round(2)
    df["y_mm"] = (df["y_mm"] - df.loc[center_idx, "y_mm"]).round(2)

# ==========================================
# 5. PLOT MAPS
# ==========================================
st.subheader("Contact Angle Maps")
map_col1, map_col2 = st.columns(2)
R = wafer_diameter_mm / 2.0

with map_col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5.5))
    ax1.add_patch(plt.Circle((0, 0), R, color="gray", fill=False, lw=2, ls="--"))
    sc = ax1.scatter(
        df["x_mm"], df["y_mm"],
        c=df["contact_angle_deg"], cmap="viridis_r",
        vmin=vmin, vmax=vmax,
        s=df["base_radius_mm"] * 40, edgecolors="black", alpha=0.9
    )
    plt.colorbar(sc, ax=ax1, extend="max", label="Water Contact Angle (°)")
    ax1.set_xlim(-R * 1.1, R * 1.1)
    ax1.set_ylim(-R * 1.1, R * 1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X Position (mm)")
    ax1.set_ylabel("Y Position (mm)")
    ax1.set_title(f"Droplet Map ({droplet_volume_ul:.0f} µL Water)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with map_col2:
    if len(df) >= 3:
        grid_res = 300
        xi = np.linspace(-R, R, grid_res)
        yi = np.linspace(-R, R, grid_res)
        X, Y = np.meshgrid(xi, yi)

        rbf = RBFInterpolator(df[["x_mm", "y_mm"]].values, df["contact_angle_deg"].values,
                              kernel="thin_plate_spline", smoothing=2.0)
        Z = rbf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        Z[X**2 + Y**2 > R**2] = np.nan

        fig2, ax2 = plt.subplots(figsize=(6, 5.5))
        im = ax2.imshow(Z, extent=[-R, R, -R, R], origin="lower",
                        cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax2.add_patch(plt.Circle((0, 0), R, color="black", fill=False, lw=2))
        plt.colorbar(im, ax=ax2, extend="max", label="Water Contact Angle (°)")
        ax2.set_xlim(-R * 1.1, R * 1.1)
        ax2.set_ylim(-R * 1.1, R * 1.1)
        ax2.set_aspect("equal")
        ax2.set_xlabel("X Position (mm)")
        ax2.set_ylabel("Y Position (mm)")
        ax2.set_title(f"Gradient Heatmap ({droplet_volume_ul:.0f} µL Water)")
        ax2.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Need at least 3 detected droplets to generate the smooth gradient heatmap.")

# ==========================================
# 6. DATA TABLE & CSV DOWNLOAD
# ==========================================
st.subheader("Droplet Data")
st.dataframe(df, use_container_width=True)
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="wafer_droplets.csv",
    mime="text/csv"
)