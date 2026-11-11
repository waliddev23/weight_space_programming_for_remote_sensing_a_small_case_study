"""Generate the 16 handcrafted conv1 filters (Claude Fable 5.1, write_run_001).
Each filter is a [3(RGB), 3, 3] kernel = (colour vector) x (spatial kernel) [+ a second term],
then rescaled so its 27 weights have std 0.118 (Kaiming fan_out) and (for non-DC filters) zero mean."""
import json, numpy as np
np.set_printoptions(precision=4, suppress=True)

# --- spatial atoms -------------------------------------------------------------
BLUR  = np.array([[1,2,1],[2,4,2],[1,2,1]], float) / 16.0          # Gaussian smoothing
LAP   = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], float) / 8.0   # centre-surround (spot / Laplacian)
SOBX  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float) / 4.0        # vertical edges
SOBY  = SOBX.T.copy()                                              # horizontal edges
DIAG1 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]], float) / 4.0        # 45-degree edge
DIAG2 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]], float) / 4.0        # 135-degree edge
HLINE = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], float) / 6.0     # horizontal ridge (crop rows)
VLINE = HLINE.T.copy()                                             # vertical ridge
DLINE = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], float) / 6.0     # diagonal ridge
CORNER= np.array([[1,1,-1],[1,1,-1],[-1,-1,-1]], float) / 9.0      # top-left bright corner vs rest
EYE   = np.array([[0,0,0],[0,1,0],[0,0,0]], float)                 # pure colour (no spatial structure)

# --- colour atoms (input channels are ImageNet-normalised R,G,B) --------------
R, G, B = np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])
LUM  = (R+G+B)/3
VARI = 2*G - R - B          # visible vegetation index direction
GR   = G - R                # green excess over red
BR   = B - R                # water: blue reflected, red absorbed
BG   = B - 0.5*(R+G)        # blue dominance (deep/clear water)
RB   = R - B                # roofs / bare soil: red-brown excess

def outer(colour, spatial): return colour[:,None,None] * spatial[None,:,:]

filters = [
  # ---- Vegetation discriminators (0-3) ----
  ("f00_VARI_smooth",   outer(VARI, BLUR),                                "Smoothed VARI (2G-R-B) over a 3x3 Gaussian: broad response to any green vegetation (Forest, Pasture, Herbaceous, crops) and negative on soil/urban."),
  ("f01_GR_centre",     outer(GR, EYE) + 0.5*outer(GR, BLUR),             "Pixel-level green-over-red excess with mild smoothing: sharper chlorophyll detector that separates bright Pasture/crop green from reddish bare fields."),
  ("f02_VARI_edge",     outer(VARI, LAP),                                 "Laplacian applied to the VARI colour direction: fires on boundaries of vegetation patches (field edges, forest borders, tree crowns)."),
  ("f03_dark_green",    -outer(LUM, BLUR) + 0.6*outer(G, EYE),            "Dark-and-green detector: negative smoothed luminance plus a green term. Forest is dark green; Pasture is bright green; urban is bright, so this orders Forest > Pasture > urban."),
  # ---- Water bodies & moisture (4-6) ----
  ("f04_BR_smooth",     outer(BR, BLUR),                                  "Smoothed blue-minus-red: water reflects blue and absorbs red; positive on River/SeaLake, negative on roofs and soil."),
  ("f05_blue_dark",     outer(BG, BLUR) - 0.5*outer(LUM, BLUR),           "Blue dominance combined with low brightness: deep/clear water (SeaLake) is dark and blue-tinted, distinguishing it from bright blue-grey industrial roofs."),
  ("f06_blue_texture",  outer(B, LAP) + 0.3*outer(G, LAP),                "Centre-surround on the blue/green channels: small on smooth water surfaces, large on ripples, shorelines and river banks; separates River (textured, banked) from SeaLake (flat)."),
  # ---- Infrastructure & road edges (7-10), colour-neutral ----
  ("f07_sobel_x",       outer(LUM, SOBX),                                 "Luminance Sobel-x: vertical edges of roads, building walls and lot boundaries (Highway, Industrial)."),
  ("f08_sobel_y",       outer(LUM, SOBY),                                 "Luminance Sobel-y: horizontal edges of roads and building rows."),
  ("f09_diag45",        outer(LUM, DIAG1),                                "Luminance 45-degree edge detector for oblique roads and plot boundaries."),
  ("f10_diag135",       outer(LUM, DIAG2),                                "Luminance 135-degree edge detector (complements filter 9 so any road orientation is covered)."),
  # ---- Agricultural field textures (11-13), Gabor-like ridges ----
  ("f11_hline",         outer(LUM, HLINE) + 0.08*outer(GR, HLINE),         "Horizontal ridge detector (bright row between dark rows) with a small green-red term: periodic crop rows and furrows of Annual/Permanent Crop."),
  ("f12_vline",         outer(LUM, VLINE) + 0.08*outer(GR, VLINE),         "Vertical ridge detector, the 90-degree partner of filter 11 for crop rows and tree lines in orchards."),
  ("f13_dline",         outer(LUM, DLINE) + 0.08*outer(RB, DLINE),         "Diagonal ridge detector with a red-brown term: diagonal furrows, ploughed-soil stripes and vineyard rows (Permanent Crop)."),
  # ---- Urban high-density / residential (14-15) ----
  ("f14_spot",          outer(LUM, LAP) + 0.1*outer(RB, LAP),             "Luminance centre-surround spot detector with a red-brown bias: bright isolated roofs on darker streets; dense Residential areas produce many such spots."),
  ("f15_corner",        outer(LUM, CORNER) + 0.1*outer(RB, CORNER),       "Corner detector (bright quadrant vs rest) with a red-roof bias: building corners and rectangular footprints typical of Residential and Industrial blocks."),
]

TARGET_STD = 0.118
W = np.zeros((16,3,3,3))
rationales = {}
for i,(name, k, why) in enumerate(filters):
    k = k.astype(float)
    k = k - k.mean()                       # zero-mean per filter (keeps the layer mean ~0)
    k = k / k.std() * TARGET_STD           # Kaiming-normal scale
    W[i] = k
    rationales[f"filter_{i}"] = f"[{name}] {why}"
W = np.round(W, 4)
print("mean %.4f std %.4f min %.3f max %.3f" % (W.mean(), W.std(), W.min(), W.max()))
json.dump({"conv1_weight": W.tolist(), "design_rationales": rationales},
          open("eurosat_wsp/runs/write_run_001/conv1_design.json","w"), indent=1)
np.save("eurosat_wsp/runs/write_run_001/conv1_design.npy", W.astype(np.float32))
for i in range(16): print(i, filters[i][0], "chan std R/G/B = %.3f %.3f %.3f" % tuple(W[i,c].std() for c in range(3)))
