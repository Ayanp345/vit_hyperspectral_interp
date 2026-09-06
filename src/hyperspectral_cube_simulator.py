"""
hyperspectral_cube_simulator.py
==================================
Generates spatially-coherent hyperspectral CUBES (H x W x Bands) for ViT
training, not just isolated pixel spectra. Reuses the same physically-
documented absorption-feature model as the earlier SAE project (chlorophyll,
red-edge, water bands, mineral Al-OH doublets), but now arranges classes into
spatially clustered regions with smooth boundaries -- mimicking a real field
scene with a healthy zone, a disease patch spreading from one corner, a
water-stress gradient, and bare-soil margins, observed at Pixxel Firefly-like
spectral resolution (135 bands, 450-950nm VNIR + partial SWIR, 5nm steps ~
matching Firefly's actual band count).

Scene classes (cube-level label = dominant/target phenomenon):
  'healthy_field'          : uniform healthy crop, minor natural variation
  'early_blight_patch'     : healthy field with a spreading early-stage
                              disease patch (severity ramps spatially)
  'water_stress_gradient'  : irrigation-gradient water stress across the field
  'mixed_landuse'          : cultural land-use patchwork (crop + bare soil +
                              small urban/road inclusion) -- the "localized
                              cultural land-use pattern" case from the prompt
"""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'spectral_sae_alpha', 'src'))
try:
    from spectral_physics_simulator import simulate_spectrum, WAVELENGTHS, WATER_VAPOR_MASK
except ImportError:
    # Fallback self-contained mini version if the earlier project's module
    # isn't on the path (keeps this project independently runnable).
    BAND_START, BAND_END, BAND_STEP = 400, 2500, 5
    WAVELENGTHS = np.arange(BAND_START, BAND_END + BAND_STEP, BAND_STEP).astype(float)
    WATER_VAPOR_MASK = ((WAVELENGTHS >= 1350) & (WAVELENGTHS <= 1420)) | \
                       ((WAVELENGTHS >= 1800) & (WAVELENGTHS <= 1950))

    def _gaussian_dip(wl, center, fwhm, depth):
        sigma = fwhm / 2.3548
        return depth * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    def _red_edge(wl, inflection=712, steepness=0.05, low=0.05, high=0.45):
        return low + (high - low) / (1 + np.exp(-steepness * (wl - inflection)))

    def simulate_spectrum(label, severity=0.0, rng=None):
        rng = rng or np.random.default_rng()
        wl = WAVELENGTHS
        spec = np.zeros_like(wl)
        if label in ("healthy_crop", "diseased_crop", "water_stressed_crop"):
            chl_depth, water_scale, cellulose_scale, re_shift = 0.35, 1.0, 0.3, 0
            if label == "diseased_crop":
                chl_depth = 0.35 * (1 - 0.6 * severity)
                cellulose_scale = 0.3 + 0.5 * severity
                re_shift = -8 * severity
            elif label == "water_stressed_crop":
                water_scale = 1.0 + 1.2 * severity
                chl_depth = 0.35 * (1 - 0.15 * severity)
            spec += _red_edge(wl, inflection=712 + re_shift)
            spec += _gaussian_dip(wl, 480, 20, chl_depth * 0.6)
            spec += _gaussian_dip(wl, 680, 25, chl_depth)
            for c, f in [(970, 30), (1200, 30), (1450, 40), (1940, 50)]:
                spec -= _gaussian_dip(wl, c, f, 0.18 * water_scale)
            spec -= _gaussian_dip(wl, 2100, 35, 0.05 * cellulose_scale)
            spec += 0.05
            spec = np.clip(spec, 0.02, 0.9)
        elif label == "bare_soil":
            spec = 0.15 + 0.00009 * (wl - 400)
        elif label == "urban_concrete":
            spec = 0.30 + 0.00001 * (wl - 400)
        illum = rng.normal(1.0, 0.03)
        noise = rng.normal(0, 0.004, size=len(wl))
        return np.clip(spec * illum + noise, 0.0, 1.0)


# Pixxel Firefly-like: ~135 usable bands after dropping water-vapor gaps.
KEEP = ~WATER_VAPOR_MASK
WL_FULL = WAVELENGTHS[KEEP]
if len(WL_FULL) > 135:
    idx = np.linspace(0, len(WL_FULL) - 1, 135).astype(int)
    FIREFLY_WAVELENGTHS = WL_FULL[idx]
else:
    FIREFLY_WAVELENGTHS = WL_FULL
N_FIREFLY_BANDS = len(FIREFLY_WAVELENGTHS)


def _resample_to_firefly(spec_full):
    full_wl = WAVELENGTHS[KEEP]
    return np.interp(FIREFLY_WAVELENGTHS, full_wl, spec_full[KEEP])


def _smooth_field(H, W, rng, scale=6.0):
    """Smooth random field via low-frequency sinusoid sum (no scipy needed)."""
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W))
    for _ in range(4):
        kx, ky = rng.uniform(-1, 1, 2) / scale
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.4, 1.0)
        field += amp * np.sin(kx * xx + ky * yy + phase)
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


SCENE_TYPES = ["healthy_field", "early_blight_patch", "water_stress_gradient", "mixed_landuse"]


def simulate_scene(scene_type, H=16, W=16, seed=None):
    """Returns cube (H, W, N_FIREFLY_BANDS) and a per-pixel class-name grid
    (for token-level interpretability ground truth) plus the scene label."""
    rng = np.random.default_rng(seed)
    field = _smooth_field(H, W, rng)
    cube = np.zeros((H, W, N_FIREFLY_BANDS))
    pixel_labels = np.empty((H, W), dtype=object)

    if scene_type == "healthy_field":
        for i in range(H):
            for j in range(W):
                cube[i, j] = _resample_to_firefly(simulate_spectrum("healthy_crop", severity=0, rng=rng))
                pixel_labels[i, j] = "healthy_crop"

    elif scene_type == "early_blight_patch":
        # disease severity increases toward one corner, early-stage only (<=0.35)
        corner_dist = np.sqrt((np.arange(H)[:, None] - 0) ** 2 + (np.arange(W)[None, :] - 0) ** 2)
        corner_dist = corner_dist / corner_dist.max()
        severity_map = np.clip((1 - corner_dist) * 0.35 + 0.1 * field, 0, 0.4)
        for i in range(H):
            for j in range(W):
                s = severity_map[i, j]
                if s > 0.06:
                    cube[i, j] = _resample_to_firefly(simulate_spectrum("diseased_crop", severity=s, rng=rng))
                    pixel_labels[i, j] = "diseased_crop"
                else:
                    cube[i, j] = _resample_to_firefly(simulate_spectrum("healthy_crop", severity=0, rng=rng))
                    pixel_labels[i, j] = "healthy_crop"

    elif scene_type == "water_stress_gradient":
        grad = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
        severity_map = np.clip(grad * 0.8 + 0.1 * field, 0, 1)
        for i in range(H):
            for j in range(W):
                s = severity_map[i, j]
                if s > 0.15:
                    cube[i, j] = _resample_to_firefly(simulate_spectrum("water_stressed_crop", severity=s, rng=rng))
                    pixel_labels[i, j] = "water_stressed_crop"
                else:
                    cube[i, j] = _resample_to_firefly(simulate_spectrum("healthy_crop", severity=0, rng=rng))
                    pixel_labels[i, j] = "healthy_crop"

    elif scene_type == "mixed_landuse":
        # blocky patchwork: crop blocks, a soil margin, a small urban inclusion
        for i in range(H):
            for j in range(W):
                if i < 3 or j < 3:
                    lbl = "bare_soil"
                elif 6 <= i < 9 and 6 <= j < 9:
                    lbl = "urban_concrete"
                else:
                    lbl = "healthy_crop"
                base = lbl if lbl != "healthy_crop" else "healthy_crop"
                spec_label = "healthy_crop" if lbl == "healthy_crop" else lbl
                cube[i, j] = _resample_to_firefly(simulate_spectrum(spec_label, severity=0, rng=rng))
                pixel_labels[i, j] = lbl
    else:
        raise ValueError(scene_type)

    return cube, pixel_labels, scene_type


def simulate_vit_dataset(n_per_class=60, H=16, W=16, seed=0):
    rng_seed = seed
    cubes, labels, pixel_label_grids = [], [], []
    for cls_idx, scene_type in enumerate(SCENE_TYPES):
        for k in range(n_per_class):
            cube, pix_labels, _ = simulate_scene(scene_type, H, W, seed=rng_seed)
            cubes.append(cube)
            labels.append(cls_idx)
            pixel_label_grids.append(pix_labels)
            rng_seed += 1
    return np.array(cubes), np.array(labels), pixel_label_grids, FIREFLY_WAVELENGTHS.copy()


if __name__ == "__main__":
    cubes, labels, grids, wl = simulate_vit_dataset(n_per_class=5, H=16, W=16)
    print("Cubes:", cubes.shape, "bands:", len(wl), "classes:", SCENE_TYPES)
