"""
train_modal.py  —  TinyNeXt ImageNet scaling study on Modal
------------------------------------------------------------
Trains all three TinyNeXt variants (M, S, T) at four dataset sizes
(50k, 100k, 150k, 200k) so you can plot accuracy vs. training-set size
and extrapolate to the full 1.2 M ImageNet-1K dataset.

Evaluation uses the standard 50,000-image ImageNet validation set
(ILSVRC/Data/CLS-LOC/val/), identical to what the authors report.

─── One-time setup ──────────────────────────────────────────────────────────
    pip install modal
    modal token new

    # Accept the competition terms at kaggle.com first, then:
    modal secret create kaggle-secret KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>

─── Deploy the persistent web endpoint (once) ───────────────────────────────
    cd classification/
    modal deploy train_modal.py

    Modal will print a URL like:
      https://sam--tinynext-imagenet-endpoint.modal.run

─── Workflow (laptop can be off after each curl) ────────────────────────────
    # Step 1 — download ImageNet to the volume (~2-4 h, only needed once)
    curl -X POST https://<url>/download

    # Step 2a — single training run
    curl -X POST https://<url>/train \\
      -H 'Content-Type: application/json' \\
      -d '{"model_name":"tinynext_m","num_samples":50000,"epochs":100}'

    # Step 2b — full scaling study: 3 models × 4 sizes = 12 parallel GPU jobs
    curl -X POST https://<url>/train_scaling \\
      -H 'Content-Type: application/json' \\
      -d '{"epochs":100}'

    # Check progress any time
    curl https://<url>/status

─── Monitor logs ────────────────────────────────────────────────────────────
    modal app logs tinynext-imagenet
"""

import json
import os
import random
import sys
import modal
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
MOUNT_DIR          = "/data"
CODE_DIR           = "/opt/classification"
APP_NAME           = "tinynext-imagenet"
KAGGLE_COMPETITION = "imagenet-object-localization-challenge"

TRAIN_SIZES  = [50_000, 100_000, 150_000, 200_000]   # scaling-study grid
MODEL_NAMES  = ["tinynext_m", "tinynext_s", "tinynext_t"]
RANDOM_SEED  = 42

# ── Modal primitives ───────────────────────────────────────────────────────────
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(f"{APP_NAME}-vol", create_if_missing=True)

# ── Container image ────────────────────────────────────────────────────────────
# Path(__file__).parent resolves to the classification/ directory regardless
# of where `modal deploy` is invoked.
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("git", "unzip", "curl")
    # PyTorch with CUDA 12.1 (matches Modal's GPU driver)
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # ML / training dependencies  (timm pinned to match requirements.txt)
    .pip_install(
        "timm==0.5.4",
        "torchmetrics",
        "tabulate~=0.9.0",
        "einops~=0.7.0",
        "numpy<2.0",
        "matplotlib",
        "datasets",
        "fastapi[standard]==0.115.0",
    )
    # Overlay the local classification/ directory so models/* and util/* are
    # importable inside the container.
    .add_local_dir(Path(__file__).parent, CODE_DIR)
)


# ── Volume path helpers ────────────────────────────────────────────────────────

def _img_train_dir() -> Path:
    return Path(f"{MOUNT_DIR}/images/train")


def _img_val_dir() -> Path:
    return Path(f"{MOUNT_DIR}/images/val")


def _splits_dir() -> Path:
    return Path(f"{MOUNT_DIR}/splits")


def _checkpoint_dir(model_name: str, num_samples: int) -> Path:
    return Path(f"{MOUNT_DIR}/checkpoints/{model_name}/{num_samples // 1000}k")


# ── Stratified sampler ─────────────────────────────────────────────────────────

def _stratified_sample(
    class_pools: dict,   # {label_int: [path_str, ...]}
    n_total: int,
) -> list:
    """
    Proportional stratified sampling across ImageNet classes.
    Takes the first `per_class` paths from each class (deterministic).
    Returns [[path, label], ...].
    """
    n_classes = len(class_pools)
    per_class = max(1, n_total // n_classes)
    result = []
    for label in sorted(class_pools):
        for path in class_pools[label][:per_class]:
            result.append([path, label])
    return result


# ── Phase 1: Download ImageNet ─────────────────────────────────────────────────

@app.function(
    cpu=4,
    memory=16_384,
    volumes={MOUNT_DIR: vol},
    image=image,
    timeout=12 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_imagenet():
    """
    Stream ILSVRC/imagenet-1k from HuggingFace and save only what we need:
      - 50k val images  → images/val/     (~5 GB)
      - 200 images/class × 1000 classes → images/train/  (~20 GB)

    Total disk: ~25 GB instead of 150 GB.
    Fully idempotent and resumable: already-saved images are detected from disk.
    """
    import time
    from collections import defaultdict
    from datasets import load_dataset

    markers   = Path(f"{MOUNT_DIR}/markers")
    splits_dir = _splits_dir()
    for d in [markers, splits_dir, _img_train_dir(), _img_val_dir()]:
        d.mkdir(parents=True, exist_ok=True)

    hf_token      = os.environ["HF_TOKEN"]
    MAX_PER_CLASS = max(TRAIN_SIZES) // 1000   # 200 images/class → supports up to 200k subset

    # ── Step 1: val images ────────────────────────────────────────────────────
    val_marker = markers / "val_done.txt"
    val_json   = splits_dir / "val.json"
    if val_marker.exists():
        print("[1/3] Val images already saved — skipping.")
    else:
        print("[1/3] Streaming 50k val images from HuggingFace…")
        val_items = []
        val_stream = load_dataset(
            "ILSVRC/imagenet-1k", split="validation",
            streaming=True, token=hf_token,
        )
        for i, ex in enumerate(val_stream):
            path = _img_val_dir() / f"{i:05d}.jpg"
            if not path.exists():
                ex["image"].convert("RGB").save(path, "JPEG", quality=95)
            val_items.append([str(path), ex["label"]])
            if (i + 1) % 10_000 == 0:
                print(f"      {i+1:,}/50000 val images saved")

        with open(val_json, "w") as f:
            json.dump(val_items, f)
        val_marker.write_text("done")
        vol.commit()
        print(f"      Saved {len(val_items):,} val images + val.json")

    # ── Step 2: train images (streaming, save only what we need) ─────────────
    train_img_marker = markers / "train_images_done.txt"
    if train_img_marker.exists():
        print("[2/3] Train images already saved — skipping.")
    else:
        # Reconstruct progress from already-saved files (supports resume)
        class_pools: dict = defaultdict(list)
        for f in sorted(_img_train_dir().iterdir()):
            if f.suffix == ".jpg":
                label = int(f.stem.split("_")[0])
                class_pools[label].append(str(f))

        filled      = sum(1 for v in class_pools.values() if len(v) >= MAX_PER_CLASS)
        total_saved = sum(len(v) for v in class_pools.values())
        if total_saved:
            print(f"[2/3] Resuming train stream: {filled}/1000 classes full, "
                  f"{total_saved:,} images already on disk")
        else:
            print(f"[2/3] Streaming train split — saving up to "
                  f"{MAX_PER_CLASS}/class × 1000 classes (~20 GB)…")

        if filled < 1000:
            train_stream = load_dataset(
                "ILSVRC/imagenet-1k", split="train",
                streaming=True, token=hf_token,
            )
            scanned = 0
            for ex in train_stream:
                label = ex["label"]
                count = len(class_pools[label])
                if count >= MAX_PER_CLASS:
                    continue

                path = _img_train_dir() / f"{label:04d}_{count:04d}.jpg"
                if not path.exists():
                    ex["image"].convert("RGB").save(path, "JPEG", quality=95)
                class_pools[label].append(str(path))

                if len(class_pools[label]) == MAX_PER_CLASS:
                    filled += 1

                scanned += 1
                if scanned % 20_000 == 0:
                    pct = filled / 10
                    print(f"      Scanned {scanned:,} | {filled}/1000 classes full ({pct:.0f}%)")
                    vol.commit()   # checkpoint every 20k so restarts lose little work

                if filled == 1000:
                    break

        total_saved = sum(len(v) for v in class_pools.values())
        print(f"      Done: {total_saved:,} images from {len(class_pools)} classes")
        train_img_marker.write_text("done")
        vol.commit()

    # ── Step 3: build train split index files ────────────────────────────────
    train_index_marker = markers / "train_index_done.txt"
    if train_index_marker.exists():
        print("[3/3] Train indices already built — skipping.")
        return

    print("[3/3] Building train split index files…")
    class_pools = defaultdict(list)
    for f in sorted(_img_train_dir().iterdir()):
        if f.suffix == ".jpg":
            label = int(f.stem.split("_")[0])
            class_pools[label].append(str(f))

    for n_samples in TRAIN_SIZES:
        items = _stratified_sample(class_pools, n_samples)
        fname = splits_dir / f"train_{n_samples // 1000}k.json"
        with open(fname, "w") as f:
            json.dump(items, f)
        print(f"      train_{n_samples // 1000}k.json: {len(items):,} images")

    train_index_marker.write_text("done")
    vol.commit()
    print("[3/3] All indices committed. Ready for training.")


# ── Phase 2: Train ─────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    volumes={MOUNT_DIR: vol},
    image=image,
    timeout=24 * 3600,
)
def train(model_name: str, num_samples: int, epochs: int = 100):
    """
    Train a TinyNeXt variant on a fixed subset of ImageNet.

    Evaluation is on the full official 50,000-image ImageNet validation set,
    matching the authors' reported numbers exactly.

    Checkpoints are saved to the volume after every epoch so training
    resumes automatically if the container restarts.
    """
    import datetime
    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    from timm.models import create_model
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler
    from timm.loss import LabelSmoothingCrossEntropy
    from timm.utils import NativeScaler, accuracy, AverageMeter
    from timm.data import create_transform

    vol.reload()

    # Add classification source to Python path and register TinyNeXt models
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    import models.menu  # noqa: F401 — registers tinynext_m/s/t with timm

    assert model_name in MODEL_NAMES, f"model_name must be one of {MODEL_NAMES}"
    assert num_samples in TRAIN_SIZES, f"num_samples must be one of {TRAIN_SIZES}"

    # ── Directories & logging ─────────────────────────────────────────────────
    ckpt_dir = _checkpoint_dir(model_name, num_samples)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path   = ckpt_dir / "training.log"
    stats_path = ckpt_dir / "stats.json"

    def log(msg: str):
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"model={model_name}  n_train={num_samples:,}  epochs={epochs}")

    # ── Load split indices (plain file paths saved during download) ───────────
    splits_dir     = _splits_dir()
    train_idx_file = splits_dir / f"train_{num_samples // 1000}k.json"
    assert train_idx_file.exists(), (
        f"Split index not found: {train_idx_file}\n"
        "Run  curl -X POST <url>/download  first."
    )
    with open(train_idx_file) as f:
        train_items = json.load(f)   # [[path, label], ...]
    with open(splits_dir / "val.json") as f:
        val_items = json.load(f)     # [[path, label], ...]  — full 50k official val

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Paper normalisation: mean=(128/255,)*3, std=(1/255,)*3
    mean = (128. / 255,) * 3
    std  = (1.   / 255,) * 3

    train_tf = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment=None,   
        interpolation="bicubic",
        re_prob=0.0,         
        re_mode="pixel",
        re_count=1,
        mean=mean,
        std=std,
    )
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class ListDataset(Dataset):
        def __init__(self, items, transform):
            self.items     = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            img = Image.open(path).convert("RGB")
            return self.transform(img), int(label)

    train_ds = ListDataset(train_items, train_tf)
    val_ds   = ListDataset(val_items,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=384, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )
    log(f"train: {len(train_ds):,} imgs ({len(train_loader)} batches/epoch)")
    log(f"val:   {len(val_ds):,} imgs ({len(val_loader)} batches)  [official ImageNet val]")

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    model  = create_model(model_name, num_classes=1000, distillation=False)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params: {n_params / 1e6:.2f}M")

    # ── Optimizer / scheduler / loss ──────────────────────────────────────────

    import argparse
    opt_cfg = argparse.Namespace(
        opt="adamw", opt_eps=1e-8, opt_betas=None,
        clip_grad=0.02, clip_mode="agc",
        momentum=0.9, weight_decay=0.025,
        lr=6e-3, sched="cosine",
        lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0,
        warmup_lr=1e-6, min_lr=1e-5,
        decay_epochs=30, warmup_epochs=5,
        cooldown_epochs=10, patience_epochs=10,
        decay_rate=0.1, epochs=epochs,
    )
    optimizer       = create_optimizer(opt_cfg, model)
    lr_scheduler, _ = create_scheduler(opt_cfg, optimizer)
    criterion       = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_scaler     = NativeScaler()
    eval_criterion  = nn.CrossEntropyLoss()

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    start_epoch = 0
    log_dict = {
        "epoch": [], "lr": [],
        "train_loss": [], "val_loss": [],
        "top1": [], "top5": [],
    }
    best_top1   = 0.0
    latest_ckpt = ckpt_dir / "checkpoint_latest.pth"

    if latest_ckpt.exists():
        log(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        loss_scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        log_dict    = ckpt["log_dict"]
        best_top1   = max(log_dict["top1"]) if log_dict["top1"] else 0.0
        log(f"Resumed at epoch {start_epoch}, best_top1={best_top1:.2f}%")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        log(f"── Epoch {epoch}/{epochs - 1} ────────────────────────")

        # Train one epoch
        model.train()
        losses     = AverageMeter()
        batch_time = AverageMeter()
        end        = time.time()

        for batch_id, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss    = criterion(outputs, targets)

            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss_scaler(
                loss, optimizer,
                clip_grad=0.02, clip_mode="agc",
                parameters=model.parameters(),
                create_graph=is_second_order,
            )

            torch.cuda.synchronize()
            losses.update(loss.detach(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_id % 100 == 0:
                eta = datetime.timedelta(
                    seconds=int(batch_time.avg * (len(train_loader) - batch_id))
                )
                log(f"  {batch_id + 1:4d}/{len(train_loader)}"
                    f"  loss: {float(losses.val):.4f}  eta: {eta}")

        global_lr   = optimizer.param_groups[0]["lr"]
        global_loss = float(losses.avg)
        log(f"* train  lr: {global_lr:.6f}  loss: {global_loss:.4f}")
        lr_scheduler.step(epoch)

        # Evaluate on the full official val set
        model.eval()
        top1m      = AverageMeter()
        top5m      = AverageMeter()
        val_losses = AverageMeter()

        with torch.no_grad():
            for images, targets in val_loader:
                images  = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = model(images)
                    vloss  = eval_criterion(output, targets)
                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                top1m.update(acc1, n=images.size(0))
                top5m.update(acc5, n=images.size(0))
                val_losses.update(vloss.detach(), n=images.size(0))

        global_top1  = float(top1m.avg)
        global_top5  = float(top5m.avg)
        global_vloss = float(val_losses.avg)
        log(f"* eval   loss: {global_vloss:.4f}"
            f"  top1: {global_top1:.2f}%  top5: {global_top5:.2f}%")

        # Record and save stats
        log_dict["epoch"].append(epoch)
        log_dict["lr"].append(global_lr)
        log_dict["train_loss"].append(global_loss)
        log_dict["val_loss"].append(global_vloss)
        log_dict["top1"].append(global_top1)
        log_dict["top5"].append(global_top5)

        with open(stats_path, "w") as f:
            json.dump(log_dict, f, indent=2)

        is_best   = global_top1 > best_top1
        best_top1 = max(best_top1, global_top1)
        log(f"* best top1: {best_top1:.2f}%")

        state = {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler":       loss_scaler.state_dict(),
            "epoch":        epoch,
            "log_dict":     log_dict,
        }
        torch.save(state, ckpt_dir / "checkpoint_latest.pth")
        if is_best:
            torch.save(state, ckpt_dir / "checkpoint_best.pth")
        vol.commit()

    log(f"Training complete. Best top1: {best_top1:.2f}%")
    return {
        "model_name":  model_name,
        "num_samples": num_samples,
        "epochs":      epochs,
        "best_top1":   best_top1,
    }


# ── Phase 3: Scaling study coordinator ────────────────────────────────────────

@app.function(
    cpu=2,
    memory=2_048,
    volumes={MOUNT_DIR: vol},
    image=image,
    timeout=24 * 3600,
)
def train_scaling_study(epochs: int = 100):
    """
    Spawn all 12 training runs (3 models × 4 sizes) as parallel Modal GPU
    jobs and collect their results into a summary JSON.
    """
    calls = []
    for model_name in MODEL_NAMES:
        for num_samples in TRAIN_SIZES:
            label = f"{model_name} @ {num_samples // 1000}k"
            print(f"Spawning {label}…")
            call = train.spawn(model_name, num_samples, epochs)
            calls.append((label, call))

    print(f"\n{len(calls)} GPU jobs spawned.  Waiting for completions…\n")
    results = []
    for label, call in calls:
        try:
            result = call.get()
            print(f"✓ {label}  best_top1={result['best_top1']:.2f}%")
            results.append(result)
        except Exception as exc:
            print(f"✗ {label}  error: {exc}")
            results.append({"label": label, "error": str(exc)})

    vol.reload()
    out = Path(f"{MOUNT_DIR}/scaling_study_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"\nAll done. Summary → {out}")
    return results


# ── Web endpoint ───────────────────────────────────────────────────────────────

@app.function(image=image, volumes={MOUNT_DIR: vol})
@modal.asgi_app()
def endpoint():
    """
    Persistent HTTPS endpoint.  Deploy once; trigger jobs from anywhere.
    """
    import fastapi
    from pydantic import BaseModel

    web = fastapi.FastAPI(title="TinyNeXt ImageNet Trainer")

    class TrainRequest(BaseModel):
        model_name:  str = "tinynext_m"
        num_samples: int = 50_000
        epochs:      int = 100

    class ScalingRequest(BaseModel):
        epochs: int = 100

    @web.post("/download")
    async def trigger_download():
        """
        Download ImageNet from Kaggle and build split indices.
        Only needs to run once (~2-4 h).  Idempotent.
        """
        call = download_imagenet.spawn()
        return {
            "status":  "download started",
            "job_id":  call.object_id,
            "note":    "Takes ~2-4 h. Only needed once.",
            "monitor": "modal app logs tinynext-imagenet",
        }

    @web.post("/train")
    async def trigger_train(req: TrainRequest):
        """
        Spawn a single training run on an A10G GPU.
        model_name ∈ {tinynext_m, tinynext_s, tinynext_t}
        num_samples ∈ {50000, 100000, 150000, 200000}
        """
        if req.model_name not in MODEL_NAMES:
            return {"error": f"model_name must be one of {MODEL_NAMES}"}
        if req.num_samples not in TRAIN_SIZES:
            return {"error": f"num_samples must be one of {TRAIN_SIZES}"}
        call = train.spawn(req.model_name, req.num_samples, req.epochs)
        return {
            "status":     "training started",
            "job_id":     call.object_id,
            "config":     req.dict(),
            "checkpoint": f"{MOUNT_DIR}/checkpoints/{req.model_name}/{req.num_samples // 1000}k/",
            "monitor":    "modal app logs tinynext-imagenet",
        }

    @web.post("/train_scaling")
    async def trigger_scaling(req: ScalingRequest):
        """Spawn all 12 training runs (3 models × 4 sizes) in parallel."""
        call = train_scaling_study.spawn(req.epochs)
        runs = [f"{m} @ {n // 1000}k" for m in MODEL_NAMES for n in TRAIN_SIZES]
        return {
            "status":  f"scaling study started ({len(runs)} parallel GPU jobs)",
            "job_id":  call.object_id,
            "runs":    runs,
            "monitor": "modal app logs tinynext-imagenet",
        }

    @web.get("/status")
    async def get_status():
        """Show download readiness, per-run training progress, and scaling summary."""
        vol.reload()
        out: dict = {}

        markers = Path(f"{MOUNT_DIR}/markers")
        out["imagenet_downloaded"] = (markers / "download_done.txt").exists()
        out["train_index_ready"]   = (markers / "train_index_done.txt").exists()

        training: dict = {}
        ckpt_root = Path(f"{MOUNT_DIR}/checkpoints")
        if ckpt_root.exists():
            for model_dir in sorted(ckpt_root.iterdir()):
                training[model_dir.name] = {}
                for size_dir in sorted(model_dir.iterdir()):
                    stats_file = size_dir / "stats.json"
                    if stats_file.exists():
                        with open(stats_file) as f:
                            stats = json.load(f)
                        training[model_dir.name][size_dir.name] = {
                            "epochs_done": len(stats.get("epoch", [])),
                            "best_top1":   max(stats["top1"]) if stats.get("top1") else None,
                            "latest_top1": stats["top1"][-1]  if stats.get("top1") else None,
                        }
        out["training"] = training

        summary_file = Path(f"{MOUNT_DIR}/scaling_study_results.json")
        if summary_file.exists():
            with open(summary_file) as f:
                out["scaling_study_summary"] = json.load(f)

        return out

    return web
