import numpy as np

def build_weights():
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    lw = std / 3.0                                   # luminance (pixel units) from normalised input
    c1w = np.zeros((16, 3, 3, 3)); c1b = np.zeros(16)
    for c in range(3):                               # ch0-2: raw R, G, B in [0,1]
        c1w[c, c, 1, 1] = std[c]; c1b[c] = mean[c]
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    d1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) / 8.0
    kernels = [sx, sx.T, d1, d1[::-1]]
    for i, k in enumerate(kernels):                  # ch3-10: |gradient| as +/- pairs
        for s, sign in enumerate((1.0, -1.0)):
            c1w[3 + 2 * i + s] = sign * lw[:, None, None] * k[None]
    c1w[11, :, 1, 1] = lw; c1b[11] = mean.mean() - 0.6   # ch11: bright pixels, relu(L-0.6)

    c2w = np.zeros((32, 16, 3, 3)); c2b = np.zeros(32)
    c2w[0, 0:3, 1, 1] = 1 / 3.0                                   # 0: L
    c2w[1, 2, 1, 1] = 1; c2w[1, 0, 1, 1] = -1; c2b[1] = 0.3       # 1: B-R (+0.3)
    c2w[2, 1, 1, 1] = 1; c2w[2, [0, 2], 1, 1] = -0.5; c2b[2] = 0.2  # 2: G-(R+B)/2 (+0.2)
    c2w[3, 3:11, 1, 1] = 1                                        # 3: texture energy T
    c2w[4, 3:11, 1, 1] = -1; c2b[4] = 0.04                        # 4: smooth blocks
    c2w[5, 3:11, 1, 1] = 1; c2b[5] = -0.09                        # 5: very textured blocks
    for i, k in enumerate((sx, -sx, sx.T, -sx.T)):                # 6-9: coarse edges of L
        c2w[6 + i, 0:3] = k[None] / 3.0
    c2w[10, 11, 1, 1] = 1                                         # 10: bright fraction
    c2w[11, 0:3, 1, 1] = -1 / 3.0; c2b[11] = 0.3                  # 11: dark blocks

    # derived features z = M f + o
    M = np.zeros((9, 32)); o = np.zeros(9)
    M[0, 0] = 1
    M[1, 1] = 1; o[1] = -0.3
    M[2, 2] = 1; o[2] = -0.2
    M[3, 3] = 1; M[4, 4] = 1; M[5, 5] = 1
    M[6, 6:10] = 1; M[6, 0] = -0.125                 # remove zero-padding border artefact
    M[7, 10] = 1; M[8, 11] = 1

    # class centroids: AnnCrop Forest Herb Highway Industr Pasture PermCrop Resid River SeaLake
    mu = np.array([
        [.44, .25, .36, .39, .48, .34, .41, .42, .32, .24],        # L
        [-.04, .12, .03, .03, .03, .08, -.03, .04, .11, .18],      # B-R
        [0, .03, .015, .005, -.015, .07, .005, -.01, .015, 0],     # green excess
        [.036, .042, .047, .07, .133, .03, .053, .102, .05, .028], # T
        [.012, .014, .005, .002, 0, .015, .003, 0, .010, .023],    # smooth
        [.004, 0, .004, .012, .055, 0, .005, .04, .006, 0],        # high texture
        [.03, .006, .018, .025, .045, .01, .02, .03, .03, .004],   # coarse edges
        [.008, 0, .003, .005, .03, 0, .004, .006, .003, 0],        # bright
        [.002, .055, .01, .005, .002, .01, .002, .002, .03, .07],  # dark
    ]).T                                                           # [10, 9]
    s = np.array([.05, .04, .02, .025, .008, .02, .015, .01, .02])
    A = mu / s ** 2                                                # [10, 9]
    fcw = A @ M
    fcb = A @ o - 0.5 * (mu ** 2 / s ** 2).sum(1)
    return {"conv1_weight": c1w, "conv1_bias": c1b,
            "conv2_weight": c2w, "conv2_bias": c2b,
            "fc_weight": fcw, "fc_bias": fcb}