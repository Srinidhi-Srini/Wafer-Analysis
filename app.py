import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

st.set_page_config(page_title="Wafer Contact Angle Mapper", layout="wide")
st.title("Silicon Wafer Contact Angle Mapper")


# ==========================================
# 1. PHYSICS
# ==========================================
def spherical_cap_angle_deg(a_mm, vol_mm3):
    """Spherical-cap geometry (no gravity). Valid when a << capillary length."""
    def residual(theta_rad):
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        v_calc = (np.pi * (a_mm ** 3) / 3.0) * ((2.0 + c) * ((1.0 - c) ** 2)) / (s ** 3)
        return v_calc - vol_mm3

    try:
        return np.degrees(brentq(residual, np.radians(0.5), np.radians(175.0)))
    except ValueError:
        return np.nan


def _yl_profile(b, a, c):
    """Integrate the axisymmetric Young-Laplace (Bashforth-Adams) equation from the
    apex (apex radius of curvature b, mm) until the profile reaches x = a (mm).
    c = rho*g/gamma in 1/mm^2.  Returns (volume_mm3, angle_rad) or None."""
    s0 = 1e-4 * b
    y0 = [s0, s0 ** 2 / (2 * b), s0 / b, 0.0]  # x, z (down from apex), phi, V

    def rhs(s, y):
        x, z, phi, _ = y
        return [np.cos(phi), np.sin(phi),
                2.0 / b + c * z - np.sin(phi) / x,
                np.pi * x ** 2 * np.sin(phi)]

    def hit(s, y):
        return y[0] - a
    hit.terminal = True
    hit.direction = 1

    sol = solve_ivp(rhs, [s0, 6 * max(a, b)], y0, events=hit, rtol=1e-8, atol=1e-10)
    if sol.t_events[0].size == 0:
        return None
    _, _, phi, vol = sol.y_events[0][0]
    return vol, phi


def young_laplace_angle_deg(a_mm, vol_mm3, c):
    """Contact angle including gravity for a drop of base radius a_mm and volume vol_mm3.
    Returns NaN if no solution is found (e.g. contact angle above ~90 deg)."""
    try:
        def f(logb):
            r = _yl_profile(np.exp(logb), a_mm, c)
            return np.nan if r is None else r[0] - vol_mm3

        grid = np.linspace(np.log(a_mm * 400), np.log(a_mm * 1.0005), 40)  # flat -> round
        prev, fprev = grid[0], f(grid[0])
        if np.isnan(fprev) or fprev > 0:
            return np.nan
        for g in grid[1:]:
            fg = f(g)
            if np.isnan(fg):
                return np.nan
            if fg > 0:
                lb = brentq(f, g, prev, xtol=1e-8)
                r = _yl_profile(np.exp(lb), a_mm, c)
                return np.degrees(r[1]) if r else np.nan
            prev = g
        return np.nan
    except Exception:
        return np.nan


def st_show_image(image, caption):
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:  # older Streamlit
        st.image(image, caption=caption, use_container_width=True)


def st_show_df(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Wafer & Droplet Settings")
    wafer_diameter_mm = st.number_input("Wafer Diameter (mm)", value=100.0, step=10.0)
    droplet_volume_ul = st.number_input("Droplet Volume (µL)", value=20.0, step=1.0)
    scale_override = st.number_input(
        "Manual scale (px per mm, 0 = auto from wafer)", value=0.0, step=0.5, min_value=0.0,
        help="Calibrate from a ruler/known feature if the wafer outline is unreliable. "
             "Contact angle scales roughly with 1/a³, so a few % scale error becomes ~3x that in angle.")

    st.header("Liquid Properties")
    water_gamma_l = st.number_input("Surface tension γ (mN/m)", value=72.8, step=0.1)
    liquid_density = st.number_input("Density ρ (kg/m³)", value=998.0, step=1.0)

    st.header("Contact Angle Model")
    model = st.radio(
        "Model",
        ["Young-Laplace (includes gravity)", "Spherical cap (no gravity)"],
        help="For water the capillary length is ~2.7 mm. Drops with base radius of that order or "
             "larger (like 20 µL drops) are gravity-flattened, so the spherical cap is not accurate.")

    st.header("Shape & Plot Settings")
    min_circ = st.slider("Min Circularity (convex hull)", 0.30, 0.95, 0.80, 0.05)
    min_sol = st.slider("Min Solidity", 0.50, 0.99, 0.92, 0.01)
    vmin, vmax = st.slider("Colorbar Range (°)", 0, 90, (10, 35))
    max_gap_mm = st.slider("Heatmap: max distance from a droplet (mm)", 3, 30, 10)

    with st.expander("Advanced: detection thresholds"):
        wafer_a_min = st.slider("Wafer purple (Lab A >)", 125, 160, 136)
        drop_s_max = st.slider("Droplet max saturation (S <)", 10, 100, 35)
        drop_a_max = st.slider("Droplet max magenta (A <)", 125, 150, 134)
        drop_v_min = st.slider("Droplet min brightness (V >)", 50, 200, 120)
        drop_v_max = st.slider("Droplet max brightness (V <)", 200, 255, 245)

# capillary constant rho*g/gamma in 1/mm^2
cap_const = (liquid_density * 9.81 / (water_gamma_l * 1e-3)) * 1e-6
cap_length_mm = 1.0 / np.sqrt(cap_const)

# ==========================================
# 3. FILE UPLOAD & PROCESSING
# ==========================================
uploaded_file = st.file_uploader("Upload a wafer image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("Upload a wafer image above to generate the contact angle maps.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image file.")
    st.stop()

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
    st.error("Wafer not detected. Try lowering 'Wafer purple (Lab A >)' under "
             "Advanced: detection thresholds in the sidebar.")
    st.stop()

hull = cv2.convexHull(np.vstack(valid_wafer_cnts))
# Ellipse fit on the hull is less sensitive to outliers than minEnclosingCircle
(wafer_cx_f, wafer_cy_f), (ax1_len, ax2_len), _ = cv2.fitEllipse(hull)
wafer_cx, wafer_cy = int(round(wafer_cx_f)), int(round(wafer_cy_f))
wafer_r_px = int(round((ax1_len + ax2_len) / 4.0))  # mean semi-axis

px_per_mm = scale_override if scale_override > 0 else wafer_r_px / (wafer_diameter_mm / 2.0)
mm_per_px = 1.0 / px_per_mm

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
vol_mm3 = droplet_volume_ul  # 1 µL = 1 mm³

for cnt in contours:
    area = cv2.contourArea(cnt)
    if not (min_area <= area <= max_area):
        continue

    hull_cnt = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_cnt)
    hull_perim = cv2.arcLength(hull_cnt, True)
    if hull_perim == 0:
        continue
    # Circularity of the convex hull: overall roundness. Notches/tails/merged blobs are
    # caught by solidity instead, so small edge defects do not reject a real droplet.
    circularity = 4 * np.pi * hull_area / (hull_perim ** 2)
    solidity = area / hull_area if hull_area > 0 else 0
    if circularity < min_circ or solidity < min_sol:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]

    # Radius from convex-hull area: robust to small notches / reflections on the edge
    r_px = np.sqrt(hull_area / np.pi)

    # Positions are already relative to the wafer centre
    x_mm = (cx_px - wafer_cx) * mm_per_px
    y_mm = -(cy_px - wafer_cy) * mm_per_px
    a_mm = r_px * mm_per_px

    theta_cap = spherical_cap_angle_deg(a_mm, vol_mm3)
    theta_yl = young_laplace_angle_deg(a_mm, vol_mm3, cap_const)

    if model.startswith("Young"):
        theta_deg = theta_yl if not np.isnan(theta_yl) else theta_cap  # fall back if no solution
    else:
        theta_deg = theta_cap
    if np.isnan(theta_deg):
        continue

    work_of_adhesion = water_gamma_l * (1 + np.cos(np.radians(theta_deg)))

    records.append({
        "droplet_id": len(records) + 1,
        "x_mm": round(x_mm, 2),
        "y_mm": round(y_mm, 2),
        "base_radius_mm": round(a_mm, 3),
        "bond_number": round(cap_const * a_mm ** 2, 2),
        "angle_spherical_cap_deg": round(theta_cap, 2),
        "angle_young_laplace_deg": round(theta_yl, 2) if not np.isnan(theta_yl) else np.nan,
        "contact_angle_deg": round(theta_deg, 2),
        "work_of_adhesion_mN_m": round(work_of_adhesion, 2),
    })

    cv2.circle(debug_img, (int(round(cx_px)), int(round(cy_px))), int(round(r_px)), (0, 0, 255), 2)
    cv2.putText(debug_img, f"{theta_deg:.1f}", (int(cx_px) - 15, int(cy_px)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

df = pd.DataFrame(records)

# ==========================================
# 4. DISPLAY DEBUG IMAGES
# ==========================================
st.subheader(f"Detection Preview ({len(df)} droplets detected)")
st.caption(f"Scale: {px_per_mm:.2f} px/mm · wafer radius {wafer_r_px} px · "
           f"capillary length for this liquid ≈ {cap_length_mm:.2f} mm")
col1, col2 = st.columns(2)
with col1:
    st_show_image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), "Detected Droplets")
with col2:
    st_show_image(droplet_mask, "Droplet Segmentation Mask")

if df.empty:
    st.warning("0 droplets detected. Open 'Advanced: detection thresholds' in the sidebar "
               "and adjust S, A and V while watching the mask.")
    st.stop()

if model.startswith("Spherical") and (df["bond_number"] > 0.5).any():
    st.warning(
        f"{(df['bond_number'] > 0.5).sum()} droplets have base radius comparable to or larger than "
        f"the capillary length ({cap_length_mm:.1f} mm), so gravity is not negligible and the "
        "spherical-cap angles are approximate. Consider the Young-Laplace model.")
if model.startswith("Young") and df["angle_young_laplace_deg"].isna().any():
    st.info("Some droplets had no Young-Laplace solution and used the spherical-cap angle instead.")

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

        pts = df[["x_mm", "y_mm"]].values
        rbf = RBFInterpolator(pts, df["contact_angle_deg"].values,
                              kernel="thin_plate_spline", smoothing=2.0)
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        Z = rbf(grid_pts).reshape(X.shape)
        Z[X ** 2 + Y ** 2 > R ** 2] = np.nan
        # Hide regions far from any measurement (the interpolation is only an extrapolation there)
        dist, _ = cKDTree(pts).query(grid_pts)
        Z[dist.reshape(X.shape) > max_gap_mm] = np.nan

        fig2, ax2 = plt.subplots(figsize=(6, 5.5))
        im = ax2.imshow(Z, extent=[-R, R, -R, R], origin="lower",
                        cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax2.add_patch(plt.Circle((0, 0), R, color="black", fill=False, lw=2))
        ax2.scatter(df["x_mm"], df["y_mm"], c="k", s=6)
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
st.caption(f"Mean angle: {df['contact_angle_deg'].mean():.1f}° · "
           f"std: {df['contact_angle_deg'].std():.1f}° · "
           f"range: {df['contact_angle_deg'].min():.1f}°–{df['contact_angle_deg'].max():.1f}°")
st_show_df(df)
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="wafer_droplets.csv",
    mime="text/csv"
)