"""Export everything the explainer page needs into site/data/.

Every number and image on the page comes from this script, which reads only result files,
checkpoints and data already in the repository. Nothing is typed by hand.

    python site/build_data.py            # all steps
    python site/build_data.py write init # selected steps

Steps: write, spectra, net, init, read, edit, pilot_write.
The steps `spectra`, `net` and `edit` need the datasets (see the README, "Data").
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
WSP = ROOT / "eurosat_wsp"
OUT = SITE / "data"
for p in (WSP, WSP / "experiment_4_write_full", WSP / "experiment_5_write_code",
          WSP / "experiment_6_write_multispectral"):
    sys.path.insert(0, str(p))

import common as wsp  # noqa: E402

W1 = WSP / "experiment_4_write_full"
W2 = WSP / "experiment_5_write_code"
W3 = WSP / "experiment_6_write_multispectral"
CLASSES = list(wsp.CLASS_NAMES)
SAMPLES_PER_CLASS = 3
SAMPLE_SEED = 2026


def rj(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wj(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"  {name}: {len(text) / 1024:.0f} kB")


def rnd(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def prose(response: Path) -> str:
    """The model's explanation: everything in its reply before the code block."""
    text = response.read_text(encoding="utf-8")
    return text.split("```")[0].strip()


def sample_indices() -> np.ndarray:
    """Test images shown on the page: picked at random, a fixed number per class."""
    y = np.load(WSP / "data" / "ms" / "test_y.npy")
    g = np.random.default_rng(SAMPLE_SEED)
    return np.concatenate([np.sort(g.choice(np.where(y == c)[0], SAMPLES_PER_CLASS, replace=False))
                           for c in range(10)])


# ── write: W1, W2, W3 scores, baselines, the model's own explanations ────────
def slim(rep: dict) -> dict:
    return {"accuracy": rep["test_accuracy"], "n": rep["n"], "correct": rep["correct"],
            "recall": [rep["per_class_recall"][c] for c in CLASSES],
            "confusion": rep["confusion_matrix"]}


def step_write() -> None:
    def rounds(folder: Path):
        out = []
        for f in sorted((folder / "private").glob("round_*.json")):
            d = rj(f)
            r = d["round"]
            out.append({"round": r, "practice": slim(d["practice"]), "test": slim(d["test"]),
                        "explanation": prose(folder / "runs" / f"round_{r}" / "response.txt")})
        return out

    b1, b2, b3 = rj(W1 / "baselines" / "baselines.json"), rj(W2 / "private" / "baselines.json"), \
        rj(W3 / "private" / "baselines.json")
    w1 = rj(W1 / "runs" / "w1_attempt_1" / "result.json")
    import w3 as w3mod
    wj("write.json", {
        "classes": CLASSES,
        "bands": [{"id": b, "name": n, "nm": nm, "res": res} for b, n, nm, res in w3mod.BANDS],
        "w1": {**slim(w1), "explanation": prose(W1 / "runs" / "w1_attempt_1" / "response.txt"),
               "baselines": {"constant": w1["majority_class_rate"], "random": b1["kaiming_mean"],
                             "centroid": b1["centroid"]["test_accuracy"], "trained": b1["trained_mean"]}},
        "w2": {"rounds": rounds(W2),
               "baselines": {"constant": w1["majority_class_rate"], "random": b2["kaiming_mean"],
                             "centroid": b1["centroid"]["test_accuracy"], "trained": b2["trained_mean"]}},
        "w3": {"rounds": rounds(W3),
               "baselines": {"constant": b3["constant"], "random": b3["kaiming_mean"],
                             "centroid": b3["centroid"], "trained": b3["trained_mean"]}},
    })


# ── spectra: what each class really looks like, and what the model assumed ───
def believed_prototypes(build_file: Path) -> np.ndarray:
    """The prototype table inside the model's build_weights(): rows = features, columns = classes."""
    captured = {}

    def prof(frame, event, _arg):
        if event == "return" and frame.f_code.co_name == "build_weights":
            captured.update(frame.f_locals)

    ns: dict = {}
    exec(compile(build_file.read_text(encoding="utf-8"), str(build_file), "exec"), ns)
    sys.setprofile(prof)
    try:
        ns["build_weights"]()
    finally:
        sys.setprofile(None)
    return np.asarray(captured["P"], dtype=np.float64)


def raw_band_features(w1: np.ndarray) -> dict[int, int]:
    """conv1 channels that simply copy one band: {band index: channel index}."""
    out = {}
    for o in range(w1.shape[0]):
        nz = np.argwhere(w1[o] != 0)
        if len(nz) == 1 and tuple(nz[0][1:]) == (1, 1) and w1[o][tuple(nz[0])] == 1:
            out[int(nz[0][0])] = o
    return out


def step_spectra() -> None:
    x = np.load(WSP / "data" / "ms" / "train_x.npy", mmap_mode="r")
    y = np.load(WSP / "data" / "ms" / "train_y.npy")
    sums, counts = np.zeros((10, 13)), np.zeros(10)
    for i in range(0, len(y), 1024):
        m = np.asarray(x[i:i + 1024]).astype(np.float64).mean(axis=(2, 3)) / 10000.0
        np.add.at(sums, y[i:i + 1024], m)
        np.add.at(counts, y[i:i + 1024], 1)
    measured = sums / counts[:, None]
    believed = []
    for r in range(4):
        rd = W3 / "runs" / f"round_{r}"
        P = believed_prototypes(rd / "build_weights.py")
        chan = raw_band_features(np.load(rd / "weights.npz")["conv1_weight"])
        believed.append({"round": r, "bands": sorted(chan),
                         "values": [rnd(P[chan[b]], 4) for b in sorted(chan)]})   # [band][class]
    wj("spectra.json", {"measured": rnd(measured, 4), "n_train": int(len(y)), "believed": believed})


# ── net: the W3 networks, and sample images to run them on in the browser ────
def step_net() -> None:
    import w3 as w3mod
    from init_compare import to_tensor
    idx = sample_indices()
    xt = np.load(WSP / "data" / "ms" / "test_x.npy", mmap_mode="r")
    yt = np.load(WSP / "data" / "ms" / "test_y.npy")
    xs = np.stack([np.asarray(xt[i]) for i in idx])                       # [N, 13, 64, 64] uint16

    nets, used = [], set()
    for r in range(4):
        w = dict(np.load(W3 / "runs" / f"round_{r}" / "weights.npz"))
        used |= set(np.where(np.abs(w["conv1_weight"]).sum(axis=(0, 2, 3)) > 0)[0].tolist())
        net = w3mod.SmallNetMS().load_written(w).eval()
        with torch.no_grad():
            logits = net(torch.from_numpy(xs.astype(np.float32) / w3mod.SCALE)).numpy()
        nets.append({"round": r, "expected_logits": rnd(logits, 3),
                     **{k: [float(np.float32(v)) for v in w[k].ravel()] for k in w3mod.SHAPES}})
    used = sorted(used)

    # the one constant used by the "rescaled" starting point in W3-init (same code path)
    xtr, _ = w3mod.load_ms("train")
    order = np.random.default_rng(0).choice(len(xtr), 1000, replace=False)
    cal = to_tensor(np.stack([np.asarray(xtr[i]) for i in order]), torch.device("cpu"), torch.float16)
    net = w3mod.SmallNetMS().load_written(dict(np.load(W3 / "runs" / "round_3" / "weights.npz"))).eval()
    with torch.no_grad():
        scale = 1.0 / torch.cat([net(cal[i:i + 256].float()) for i in range(0, 1000, 256)]).std().item()
        # how often the network is more than 99% sure of its answer, on the whole test set
        lt = torch.cat([net(torch.from_numpy(np.asarray(xt[i:i + 512]).astype(np.float32) / w3mod.SCALE))
                        for i in range(0, len(yt), 512)])
        sure = {"as_written": float((torch.softmax(lt, 1).max(1).values > 0.99).float().mean()),
                "rescaled": float((torch.softmax(lt * scale, 1).max(1).values > 0.99).float().mean())}

    OUT.mkdir(parents=True, exist_ok=True)
    xs[:, used].astype("<u2").tofile(OUT / "samples.bin")
    print(f"  samples.bin: {(OUT / 'samples.bin').stat().st_size / 1e6:.1f} MB "
          f"({len(idx)} images x {len(used)} bands)")

    from PIL import Image

    def sprite(bands, name):
        img = xs[:, bands].astype(np.float32) / 10000.0                     # [N, 3, 64, 64]
        lo = np.percentile(img, 1, axis=(0, 2, 3), keepdims=True)           # per-channel stretch
        hi = np.percentile(img, 99.5, axis=(0, 2, 3), keepdims=True)
        img = (np.clip((img - lo) / (hi - lo), 0, 1) ** 0.8 * 255).astype(np.uint8)
        sheet = np.zeros((SAMPLES_PER_CLASS * 64, 10 * 64, 3), dtype=np.uint8)
        for k in range(len(idx)):
            c, j = divmod(k, SAMPLES_PER_CLASS)
            sheet[j * 64:(j + 1) * 64, c * 64:(c + 1) * 64] = img[k].transpose(1, 2, 0)
        Image.fromarray(sheet).save(OUT / name, optimize=True)
        return [lo.ravel().tolist(), hi.ravel().tolist()]

    stretch = {"rgb": sprite([3, 2, 1], "samples_rgb.png"), "nir": sprite([7, 3, 2], "samples_nir.png")}
    wj("net.json", {
        "sample_seed": SAMPLE_SEED, "per_class": SAMPLES_PER_CLASS,
        "test_index": idx.tolist(), "labels": yt[idx].tolist(), "used_bands": used,
        "mean_spectrum": rnd(xs.astype(np.float64).mean(axis=(2, 3)) / 10000.0, 4),
        "stretch": stretch, "rescale": scale, "sure": sure, "rounds": nets,
    })


# ── init: W3-init ────────────────────────────────────────────────────────────
def step_init() -> None:
    res = rj(W3 / "init_compare" / "results.json")
    runs = [{"budget": r["budget"], "n_train": r["n_train"], "seed": r["seed"], "init": r["init"],
             "epochs": r["epochs"], "test": r["test_accuracy"], "val": rnd(r["val_history"], 4)}
            for r in res.values()]
    before = rj(W3 / "private" / "round_3.json")["test"]["test_accuracy"]
    wj("init.json", {"runs": runs, "fable_untrained": before})


# ── read: the nine pilot models ──────────────────────────────────────────────
def step_read() -> None:
    from read_features import correlation_matrix, fingerprint
    key = rj(WSP / "prompts" / "key_read_exp1_all9_models.json")["models"]
    res = rj(WSP / "experiment_1_read" / "metrics" / "llm_read_9models_result.json")
    models, conv1, heads = [], [], []
    for name in sorted(key):
        sd = wsp.load_state_dict(WSP / "experiment_1_read" / "checkpoints" / key[name]["source_file"])
        conv1.append(sd["conv1.weight"])
        heads.append(sd["classifier.2.weight"])
        models.append({"id": name[-2:], "condition": key[name]["raw_condition"], "seed": key[name]["train_seed"],
                       "predicted": key[next(k for k in key if key[k]["condition"] == res["details"][name]["predicted"])]["raw_condition"],
                       "correct": res["details"][name]["correct"], **{k: round(v, 4) for k, v in fingerprint(sd).items()}})
    wj("read.json", {"models": models, "conv1_corr": rnd(correlation_matrix(conv1), 3),
                     "head_corr": rnd(correlation_matrix(heads), 3)})


# ── edit: the damaged ResNet-18 ──────────────────────────────────────────────
def step_edit() -> None:
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet18
    from edit_repairs import mean_filter, zero_rows
    layer = "layer3.0.conv1.weight"
    E = WSP / "experiment_2_edit"
    original = wsp.load_state_dict(E / "original" / "resnet18_original_state_dict.pt")
    damaged = wsp.load_state_dict(E / "corrupted" / "resnet18_corrupted_a_gaussian_noise.pt")
    ns: dict = {}
    script = WSP / "runs" / "edit_run_001" / "parsed_output" / "repair_script.py"
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), ns)
    states = {"original": original[layer], "damaged": damaged[layer],
              "repaired": ns["repair_layer3_conv1"](damaged[layer]), "zeroed": zero_rows(damaged[layer])}
    assert torch.allclose(states["repaired"], mean_filter(damaged[layer]), atol=1e-6)

    # each filter (1,152 weights) as a 32 x 36 tile, 16 x 16 tiles, one grey scale for all three images
    healthy_sd = float(original[layer].std())
    lim = 8 * healthy_sd
    for name in ("original", "damaged", "repaired"):
        M = states[name].reshape(256, 32, 36).numpy()
        sheet = np.full((16 * 34 - 2, 16 * 38 - 2), 255, dtype=np.uint8)
        alpha = np.zeros_like(sheet)
        for f in range(256):
            r, c = divmod(f, 16)
            sheet[r * 34:r * 34 + 32, c * 38:c * 38 + 36] = (np.clip(M[f] / lim, -1, 1) * 127 + 128).astype(np.uint8)
            alpha[r * 34:r * 34 + 32, c * 38:c * 38 + 36] = 255
        Image.fromarray(np.dstack([sheet, alpha]), "LA").save(OUT / f"edit_{name}.png", optimize=True)

    # behaviour: the whole test set, and the page's sample images
    device = wsp.get_device()
    raw = wsp.load_raw()
    x, y = wsp.preload_split(raw["test"])
    assert (y == np.load(WSP / "data" / "ms" / "test_y.npy")).all(), "RGB and multispectral test sets differ"
    # the harness's own evaluation pipeline (wsp_eval.evaluate_edit), so the recorded scores are reproduced
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(wsp.MEAN, wsp.STD)])
    loader = wsp.make_loader(wsp.ArrayDataset(x, y, tf), batch_size=128, shuffle=False)
    idx = sample_indices()
    behaviour = {}
    for name in ("original", "damaged", "repaired", "zeroed"):
        model = resnet18(num_classes=10)
        sd = dict(original)
        sd[layer] = states[name]
        model.load_state_dict(sd)
        model.to(device).eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    probs.append(torch.softmax(model(xb.to(device)).float(), 1).cpu())
        probs = torch.cat(probs).numpy()
        pred = probs.argmax(1)
        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (y, pred), 1)
        behaviour[name] = {"accuracy": float((pred == y).mean()), "confusion": cm.tolist(),
                           "sample_pred": pred[idx].tolist(), "sample_conf": rnd(probs[idx].max(1), 3)}
        print(f"  ResNet-18 {name}: test {behaviour[name]['accuracy'] * 100:.2f}%")

    base = rj(E / "metrics" / "edit_baselines.json")
    recorded = rj(E / "metrics" / "llm_repair_a_result.json")
    wj("edit.json", {
        "layer": layer, "shape": list(original[layer].shape), "healthy_sd": healthy_sd,
        "row_std": {k: rnd(states[k].reshape(256, -1).std(dim=1), 4) for k in ("original", "damaged", "repaired")},
        "behaviour": behaviour,
        "recorded": {"original": recorded["original_accuracy"], "damaged": recorded["corrupted_accuracy"],
                     "repaired": recorded["repaired_accuracy"]},
        "methods": {v: {"name": d["name"], "damaged": d["corrupted_test_accuracy"],
                        "methods": {m: r["test_accuracy"] for m, r in d["methods"].items()}}
                    for v, d in base["variants"].items()},
        "original_accuracy": base["original_test_accuracy"],
    })


# ── pilot write: the hand-written first layer ────────────────────────────────
def step_pilot_write() -> None:
    R = WSP / "runs" / "write_run_001"
    W = torch.load(R / "parsed_output" / "first_conv_fable.pt", map_location="cpu", weights_only=False)
    W = W["conv1.weight"] if isinstance(W, dict) else W
    design = rj(R / "conv1_design.json")["design_rationales"]
    notes = []
    for i in range(16):
        m = re.match(r"\[(\w+)\]\s*(.*)", design[f"filter_{i}"], flags=re.S)
        notes.append({"tag": m.group(1), "text": m.group(2)})
    runs = []
    for f in sorted((WSP / "experiment_3_write" / "trained_runs").glob("first_conv_*.json")):
        d = rj(f)
        runs.append({"init": "fable" if "fable" in f.name else "kaiming", "seed": d["seed"], "test": d["test_acc"],
                     "val": rnd([h["val_acc"] for h in d["history"]], 4)})
    wj("pilot_write.json", {"filters": rnd(W.numpy(), 4), "notes": notes, "runs": runs})


STEPS = {"write": step_write, "spectra": step_spectra, "net": step_net, "init": step_init,
         "read": step_read, "edit": step_edit, "pilot_write": step_pilot_write}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(STEPS)):
        print(name)
        STEPS[name]()
