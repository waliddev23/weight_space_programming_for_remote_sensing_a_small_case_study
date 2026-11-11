"""Parse the 9-model READ prompt and compute weight-space fingerprint features.
Reads ONLY the prompt text (no key, no checkpoint dir)."""
import re, sys, json
import numpy as np

PROMPT = r"D:\PhD_Thesis\UM6P\presto\weight_space_programming\eurosat_wsp\prompts\prompt_read_exp1_all9_models.txt"
txt = open(PROMPT, encoding="utf-8").read()

# split into model blocks
blocks = re.split(r"^=== (Model_\d\d) ===$", txt, flags=re.M)
models = {}
for i in range(1, len(blocks), 2):
    name, body = blocks[i], blocks[i+1]
    W = np.zeros((16,3,3,3), dtype=np.float64)
    for m in re.finditer(r"conv1\.weight filter (\d+) chan ([RGB]) row (\d)\s+(-?[\d.]+) (-?[\d.]+) (-?[\d.]+)", body):
        f, c, r = int(m.group(1)), "RGB".index(m.group(2)), int(m.group(3))
        W[f, c, r, :] = [float(m.group(k)) for k in (4,5,6)]
    b = np.array([float(x) for x in re.search(r"^conv1\.bias\s+(.*)", body, re.M).group(1).split()])
    C = np.zeros((10,128))
    for m in re.finditer(r"^classifier\.2\.weight row (\d+) \([^)]*\)\s+(.*)", body, re.M):
        C[int(m.group(1))] = [float(x) for x in m.group(2).split()]
    cb = np.array([float(x) for x in re.search(r"^classifier\.2\.bias\s+(.*)", body, re.M).group(1).split()])
    models[name] = dict(W=W, b=b, C=C, cb=cb)

def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

SOBX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
SOBY = SOBX.T
LAP  = np.array([[0,1,0],[1,-4,1],[0,1,0]], float)
D1   = np.array([[0,1,2],[-1,0,1],[-2,-1,0]], float)
D2   = np.array([[2,1,0],[1,0,-1],[0,-1,-2]], float)

def features(W, b, C, cb):
    F = {}
    F["conv1_fro"] = float(np.linalg.norm(W))
    F["conv1_std"] = float(W.std())
    F["conv1_absmax"] = float(np.abs(W).max())
    F["conv1_bias_norm"] = float(np.linalg.norm(b))
    fn = np.linalg.norm(W.reshape(16,-1), axis=1)
    F["conv1_filter_norm_cv"] = float(fn.std()/fn.mean())
    F["conv1_filter_norm_min"] = float(fn.min())
    # --- rotational closure of the filter bank: for each filter and each 90deg rotation / flip,
    #     max cosine similarity to any filter in the bank (excluding identity transform of itself)
    bank = W.reshape(16, -1)
    rot_scores, flip_scores = [], []
    for f in range(16):
        for k in (1,2,3):
            Wr = np.rot90(W[f], k=k, axes=(1,2)).ravel()
            sims = [cos(Wr, bank[g]) for g in range(16)]
            rot_scores.append(max(sims))
        for ax in (1,2):
            Wf = np.flip(W[f], axis=ax).ravel()
            sims = [cos(Wf, bank[g]) for g in range(16)]
            flip_scores.append(max(sims))
    F["rot90_closure"] = float(np.mean(rot_scores))
    F["flip_closure"] = float(np.mean(flip_scores))
    # self rotational symmetry (filter vs its own rotations)
    F["self_rot_sym"] = float(np.mean([cos(np.rot90(W[f],k,axes=(1,2)), W[f]) for f in range(16) for k in (1,2,3)]))
    # --- anisotropy: horizontal vs vertical derivative energy per channel-kernel
    aniso, lap_frac, center_frac = [], [], []
    for f in range(16):
        for c in range(3):
            k = W[f,c]
            ex = (k*SOBX).sum()**2; ey = (k*SOBY).sum()**2
            e1 = (k*D1).sum()**2; e2 = (k*D2).sum()**2
            tot = ex+ey+e1+e2+1e-12
            aniso.append(abs(ex-ey)/(ex+ey+1e-12))
            lap_frac.append(abs((k*LAP).sum())/ (np.abs(k).sum()+1e-12))
            center_frac.append(k[1,1]**2 / ((k**2).sum()+1e-12))
    F["aniso_hv"] = float(np.mean(aniso))
    F["lap_resp"] = float(np.mean(lap_frac))
    F["center_energy_frac"] = float(np.mean(center_frac))
    # axis-aligned edge-ness: energy of best of (Sobel x, Sobel y) relative to best diagonal
    axis_e, diag_e = [], []
    for f in range(16):
        k = W[f].sum(0)  # luminance kernel
        axis_e.append(max((k*SOBX).sum()**2, (k*SOBY).sum()**2))
        diag_e.append(max((k*D1).sum()**2, (k*D2).sum()**2))
    F["axis_vs_diag"] = float(np.mean(axis_e)/ (np.mean(diag_e)+1e-12))
    # --- channel divergence: luminance fraction and inter-channel cosine
    lum_frac, ch_cos = [], []
    for f in range(16):
        k = W[f].reshape(3, 9)
        mean_k = k.mean(0, keepdims=True)
        lum_frac.append((mean_k**2).sum()*3 / ((k**2).sum()+1e-12))
        ch_cos.append(np.mean([cos(k[0],k[1]), cos(k[0],k[2]), cos(k[1],k[2])]))
    F["lum_frac"] = float(np.mean(lum_frac))
    F["chan_cos"] = float(np.mean(ch_cos))
    F["chan_cos_med"] = float(np.median(ch_cos))
    # channel-wise std
    F["std_R"], F["std_G"], F["std_B"] = [float(W[:,c].std()) for c in range(3)]
    F["R_minus_B_energy"] = float(((W[:,0]-W[:,2])**2).sum() / (W**2).sum())
    # --- classifier
    F["cls_fro"] = float(np.linalg.norm(C))
    F["cls_std"] = float(C.std())
    F["cls_absmax"] = float(np.abs(C).max())
    s = np.linalg.svd(C, compute_uv=False)
    F["cls_spec"] = float(s[0]); F["cls_s_ratio"] = float(s[0]/s[-1])
    p = s/s.sum(); F["cls_eff_rank"] = float(np.exp(-(p*np.log(p)).sum()))
    F["cls_bias_norm"] = float(np.linalg.norm(cb)); F["cls_bias_range"] = float(cb.max()-cb.min())
    rn = np.linalg.norm(C, axis=1); F["cls_row_norm_cv"] = float(rn.std()/rn.mean())
    # row cosine similarity (class prototypes)
    Cn = C/np.linalg.norm(C,axis=1,keepdims=True)
    G = Cn@Cn.T; F["cls_row_cos_mean"] = float((G.sum()-10)/90)
    # column (feature) usage: fraction of 128 features with tiny column norm
    cn = np.linalg.norm(C, axis=0); F["cls_col_norm_cv"] = float(cn.std()/cn.mean())
    F["cls_kurtosis"] = float(((C-C.mean())**4).mean()/C.var()**2)
    F["conv1_kurtosis"] = float(((W-W.mean())**4).mean()/W.var()**2)
    return F

allF = {n: features(**models[n]) for n in sorted(models)}
keys = list(next(iter(allF.values())).keys())
print(f"{'feature':24s}" + "".join(f"{n[-2:]:>9s}" for n in sorted(allF)))
for k in keys:
    row = [allF[n][k] for n in sorted(allF)]
    print(f"{k:24s}" + "".join(f"{v:9.3f}" for v in row))
# rank table (1 = lowest)
print("\nRANKS (1=lowest)")
print(f"{'feature':24s}" + "".join(f"{n[-2:]:>9s}" for n in sorted(allF)))
for k in keys:
    row = np.array([allF[n][k] for n in sorted(allF)])
    r = row.argsort().argsort()+1
    print(f"{k:24s}" + "".join(f"{v:9d}" for v in r))
json.dump(allF, open(sys.argv[1], "w"), indent=1)
np.save(sys.argv[1].replace(".json","_W.npy"), np.stack([models[n]["W"] for n in sorted(models)]))
np.save(sys.argv[1].replace(".json","_C.npy"), np.stack([models[n]["C"] for n in sorted(models)]))
