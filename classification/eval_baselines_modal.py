"""
eval_baselines_modal.py — Evaluate comparison models on ImageNet-1K val set
----------------------------------------------------------------------------
Models evaluated:
  1. MobileOne-S0       (Apple)           — reparametrised before eval
  2. MobileNet V1 1.0   (timm pretrained)
  3. PVTv2-B0           (Wang et al.)
  4. EfficientViT-M2    (Microsoft Cream)
  5. ShuffleNet V2 x1.0 (torchvision)

All models are evaluated on the full official 50k ImageNet-1K val set
stored in /data/images/val/ (written by train_modal.py's download step).

─── Deploy & run ────────────────────────────────────────────────────────────
    cd classification/
    modal deploy eval_baselines_modal.py

    # Trigger all 5 evals in parallel (~10-20 min total)
    curl -X POST https://<url>/eval

    # Fetch results once done
    curl https://<url>/results

─── Monitor ─────────────────────────────────────────────────────────────────
    modal app logs tinynext-eval-baselines
"""

import json
import os
import sys
import modal
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
MOUNT_DIR    = "/data"
APP_NAME     = "tinynext-eval-baselines"
VOL_NAME     = "tinynext-imagenet-vol"   # reuse the volume from training
RESULTS_PATH = f"{MOUNT_DIR}/eval_baselines.json"

# Standard ImageNet normalisation used by all baseline models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Modal primitives ───────────────────────────────────────────────────────────
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)

image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "timm",              
        "numpy<2.0",
        "Pillow",
        "einops",
        "requests",
        "fastapi[standard]==0.115.0",
    )
)


# ── Shared helpers (run inside container) ─────────────────────────────────────

def _download(url: str, dest: str):
    """Download url → dest, following redirects (works for GitHub releases)."""
    import requests
    print(f"  Downloading {url.split('/')[-1]} …")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"  Saved → {dest}")


def _build_val_loader(batch_size: int = 256):
    """DataLoader over the 50k val JPEGs saved by download_imagenet()."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    val_json = Path(f"{MOUNT_DIR}/splits/val.json")
    assert val_json.exists(), (
        "val.json not found — run the /download endpoint in train_modal.py first."
    )
    with open(val_json) as f:
        val_items = json.load(f)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class ValDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            path, label = self.items[i]
            return self.transform(Image.open(path).convert("RGB")), int(label)

    return DataLoader(
        ValDataset(val_items, tf),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
    )


def _run_eval(model, model_name: str) -> dict:
    """Top-1 / Top-5 accuracy over the full 50k val set."""
    import torch

    device = torch.device("cuda")
    model  = model.to(device).eval()
    loader = _build_val_loader()

    correct1 = correct5 = total = 0

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(images)
                # Some models return dicts or tuples
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict):
                    out = out["logits"] if "logits" in out else next(iter(out.values()))

            _, pred5     = out.topk(5, dim=1, largest=True, sorted=True)
            tgt_exp      = targets.view(-1, 1).expand_as(pred5)
            correct5    += pred5.eq(tgt_exp).any(dim=1).sum().item()
            correct1    += pred5[:, :1].eq(tgt_exp[:, :1]).sum().item()
            total       += targets.size(0)

            if (step + 1) % 50 == 0:
                print(f"  [{model_name}] {total:,}/50000  "
                      f"top1={100.*correct1/total:.2f}%  "
                      f"top5={100.*correct5/total:.2f}%")

    top1 = round(100.0 * correct1 / total, 2)
    top5 = round(100.0 * correct5 / total, 2)
    print(f"\n[{model_name}]  top1={top1}%  top5={top5}%  (n={total:,})")
    return {"model": model_name, "top1": top1, "top5": top5, "n_val": total}


# ── Per-model eval functions ───────────────────────────────────────────────────

@app.function(
    gpu="A10G", cpu=4, memory=16_384,
    volumes={MOUNT_DIR: vol}, image=image, timeout=2 * 3600,
)
def eval_mobileone_s0() -> dict:
    """MobileOne-S0: clone Apple repo, load unfused weights, reparametrize."""
    import subprocess, torch

    print("=== MobileOne-S0 ===")

    # Clone Apple ml-mobileone repo
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/apple/ml-mobileone", "/tmp/mobileone"],
        check=True, capture_output=True,
    )
    sys.path.insert(0, "/tmp/mobileone")
    from mobileone import mobileone, reparameterize_model   # noqa

    # Download unfused weights
    wpath = "/tmp/mobileone_s0_unfused.pth.tar"
    _download(
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar",
        wpath,
    )

    model = mobileone(variant="s0")
    ckpt  = torch.load(wpath, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    # Strip 'module.' prefix added by DataParallel if present
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    print("Reparametrizing …")
    model = reparameterize_model(model)

    vol.reload()
    return _run_eval(model, "mobileone_s0")


@app.function(
    gpu="A10G", cpu=4, memory=16_384,
    volumes={MOUNT_DIR: vol}, image=image, timeout=2 * 3600,
)
def eval_mobilenet_v1() -> dict:
    """MobileNet V1 1.0 — pretrained weights via timm."""
    import timm

    print("=== MobileNet V1 1.0 ===")

    for name in ("mobilenetv1_100.ra4_e3600_r224_in1k", "mobilenetv1_100"):
        try:
            model = timm.create_model(name, pretrained=True)
            print(f"  Loaded timm model: {name}")
            break
        except Exception as exc:
            print(f"  {name} failed ({exc}), trying next …")
    else:
        raise RuntimeError("Could not load MobileNet V1 from timm")

    vol.reload()
    return _run_eval(model, "mobilenet_v1_1.0")


@app.function(
    gpu="A10G", cpu=4, memory=16_384,
    volumes={MOUNT_DIR: vol}, image=image, timeout=2 * 3600,
)
def eval_pvtv2_b0() -> dict:
    """PVTv2-B0 — load directly from timm (has pretrained weights built-in)."""
    import timm

    print("=== PVTv2-B0 ===")
    model = timm.create_model("pvt_v2_b0", pretrained=True)
    print("  Loaded pvt_v2_b0 from timm")

    vol.reload()
    return _run_eval(model, "pvtv2_b0")


@app.function(
    gpu="A10G", cpu=4, memory=16_384,
    volumes={MOUNT_DIR: vol}, image=image, timeout=2 * 3600,
)
def eval_efficientvit_m2() -> dict:
    """EfficientViT-M2: download model/*.py, import to register with timm, create model."""
    import timm, torch, importlib, requests

    print("=== EfficientViT-M2 ===")

    # Fetch the list of files in model/ from GitHub API (
    api_url = ("https://api.github.com/repos/microsoft/Cream"
               "/contents/EfficientViT/classification/model")
    entries = requests.get(api_url, timeout=30).json()

    os.makedirs("/tmp/evit/model", exist_ok=True)
    for entry in entries:
        if entry["name"].endswith(".py"):
            _download(entry["download_url"], f"/tmp/evit/model/{entry['name']}")
            print(f"  Downloaded model/{entry['name']}")

  
    sys.path.insert(0, "/tmp/evit")


    for fname in sorted(os.listdir("/tmp/evit/model")):
        if fname.endswith(".py"):
            mod_name = f"model.{fname[:-3]}"
            try:
                importlib.import_module(mod_name)
                print(f"  Imported {mod_name}")
            except Exception as e:
                print(f"  Skipped {mod_name}: {e}")

    # Download weights
    wpath = "/tmp/efficientvit_m2.pth"
    _download(
        "https://github.com/xinyuliu-jeffrey/EfficientViT_Model_Zoo/releases/download/v1.0/efficientvit_m2.pth",
        wpath,
    )

    factory = None

    # Strategy 1: search every model.* module loaded above
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and "model" in mod_name and hasattr(mod, "EfficientViT_M2"):
            factory = getattr(mod, "EfficientViT_M2")
            print(f"  Found EfficientViT_M2 in sys.modules['{mod_name}']")
            break

    # Strategy 2: timm's internal entrypoint registry
    if factory is None:
        try:
            from timm.models._registry import _model_entrypoints
            factory = _model_entrypoints.get("EfficientViT_M2")
            if factory:
                print("  Found EfficientViT_M2 in timm._model_entrypoints")
        except Exception as e:
            print(f"  timm registry fallback failed: {e}")

    if factory is None:
        summary = {
            k: [n for n in dir(v) if "EffVit" in n or "M2" in n or "efficientvit" in n.lower()]
            for k, v in sys.modules.items()
            if v is not None and "model" in k
        }
        raise RuntimeError(f"EfficientViT_M2 not found anywhere.\nSearch summary: {summary}")

    model = factory()
    ckpt = torch.load(wpath, map_location="cpu")
    model.load_state_dict(ckpt["model"])  
    print("  Loaded EfficientViT_M2 with official weights")

    vol.reload()
    return _run_eval(model, "efficientvit_m2")


@app.function(
    gpu="A10G", cpu=4, memory=16_384,
    volumes={MOUNT_DIR: vol}, image=image, timeout=2 * 3600,
)
def eval_shufflenet_v2() -> dict:
    """ShuffleNet V2 x1.0 — torchvision pretrained."""
    from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

    print("=== ShuffleNet V2 x1.0 ===")
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    vol.reload()
    return _run_eval(model, "shufflenet_v2_x1_0")


# ── Coordinator ────────────────────────────────────────────────────────────────

@app.function(
    cpu=2, memory=2_048,
    volumes={MOUNT_DIR: vol}, image=image, timeout=4 * 3600,
)
def eval_all() -> list:
    """Spawn all 5 eval jobs in parallel, collect results, save JSON."""
    import datetime

    jobs = [
        ("mobileone_s0",    eval_mobileone_s0.spawn()),
        ("mobilenet_v1",    eval_mobilenet_v1.spawn()),
        ("pvtv2_b0",        eval_pvtv2_b0.spawn()),
        # ("efficientvit_m2", eval_efficientvit_m2.spawn()),
        ("shufflenet_v2",   eval_shufflenet_v2.spawn()),
    ]
    print(f"Spawned {len(jobs)} parallel GPU eval jobs …\n")

    results = []
    for name, call in jobs:
        try:
            result = call.get(timeout=2 * 3600)
            print(f"  ✓ {name:20s}  top1={result['top1']:.2f}%  top5={result['top5']:.2f}%")
            results.append(result)
        except Exception as exc:
            print(f"  ✗ {name:20s}  ERROR: {exc}")
            results.append({"model": name, "error": str(exc)})

    # Pretty summary table
    print("\n" + "=" * 55)
    print(f"{'Model':<22} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 55)
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<20}  {'ERROR':>8}")
        else:
            print(f"  {r['model']:<20}  {r['top1']:>7.2f}%  {r['top5']:>7.2f}%")
    print("=" * 55)

    # Persist to volume
    vol.reload()
    out = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "results":   results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"\nSaved → {RESULTS_PATH}")
    return results


# ── Web endpoint ───────────────────────────────────────────────────────────────

@app.function(image=image, volumes={MOUNT_DIR: vol})
@modal.asgi_app()
def endpoint():
    import fastapi

    web = fastapi.FastAPI(title="TinyNeXt Baseline Evaluator")

    @web.post("/eval")
    async def trigger_eval():
        """Spawn all 5 eval jobs in parallel."""
        call = eval_all.spawn()
        return {
            "status":  "eval started — 5 parallel A10G GPU jobs",
            "job_id":  call.object_id,
            "monitor": f"modal app logs {APP_NAME}",
            "results_endpoint": "/results",
        }


    @web.get("/results")
    async def get_results():
        """Return saved results once eval_all has completed."""
        vol.reload()
        p = Path(RESULTS_PATH)
        if not p.exists():
            return {"status": "no results yet — POST /eval to start"}
        with open(p) as f:
            return json.load(f)

    return web
