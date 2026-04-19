# Image Classification

## Author Results (full ImageNet-1K, 100 epochs)

| Model | Top-1 | Top-5 | #Params | MACs | Latency | Logs |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TinyNeXt-M | 75.3% | 92.2% | 2.3M | 475M | 19.4ms | [log](logs/tinynext_m/rank0.log) |
| TinyNeXt-S | 72.7% | 90.9% | 1.3M | 304M | 14.3ms | [log](logs/tinynext_s/rank0.log) |
| TinyNeXt-T | 71.5% | 90.2% | 1.0M | 259M | 12.7ms | [log](logs/tinynext_t/rank0.log) |

Latency measured on Nvidia Jetson Nano.

---

## Scaling Study (Modal cloud, 50 epochs, 1× A10G)

Training TinyNeXt-T on ImageNet subsets to study data-scaling behaviour and
extrapolate performance to the full dataset.

### TinyNeXt-T results

| Train size | Top-1 | Top-5 |
|:---:|:---:|:---:|
| 50k  | 28.67% | 52.27% |
| 100k | 40.52% | 65.13% |
| 150k | 46.31% | 70.76% |
| 200k | 51.81% | 75.71% |
| **1.28M (extrapolated)** | **~66%** | **~86%** |

Extrapolation uses a saturating (Michaelis-Menten) fit with 90% bootstrap CI.
See [`plot_scaling.py`](plot_scaling.py) and [`scaling_curve.png`](scaling_curve.png).

### Baseline model comparison (50k ImageNet val, 1× A10G)

| Model | Top-1 | Top-5 | Source |
|:---:|:---:|:---:|:---:|
| MobileOne-S0 | — | — | Apple (reparametrised) |
| MobileNet V1 1.0 | — | — | timm pretrained |
| PVTv2-B0 | — | — | timm pretrained |
| EfficientViT-M2 | — | — | Microsoft Cream |
| ShuffleNet V2 x1.0 | — | — | torchvision pretrained |

Run the baseline eval (instructions below) to fill in these numbers.

---

## Reproduction Guide

### Prerequisites

```sh
pip install modal
modal token new          # authenticate with Modal
```

Create a HuggingFace secret (needed for the ImageNet download only):

```sh
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
```

Your HuggingFace account must have access to
[ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
(request access on the dataset page — approved instantly).

---

### Step 1 — Deploy the training endpoint

```sh
cd classification/
modal deploy train_modal.py
# Modal prints: https://<your-username>--tinynext-imagenet-endpoint.modal.run
```

Set the URL as a shell variable for the commands below:

```sh
TRAIN_URL="https://<your-username>--tinynext-imagenet-endpoint.modal.run"
```

---

### Step 2 — Download ImageNet to the Modal volume

Streams only what is needed (~25 GB): up to 200 images/class for training and
the full 50k official validation set. Fully resumable — safe to re-run.

```sh
curl -X POST $TRAIN_URL/download
# Monitor: modal app logs tinynext-imagenet
# Takes ~1-2 hours on first run.
```

---

### Step 3 — Train TinyNeXt-T on dataset subsets

#### Single run

```sh
curl -X POST $TRAIN_URL/train \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"tinynext_t","num_samples":200000,"epochs":50}'
```

Valid values: `model_name` ∈ `{tinynext_m, tinynext_s, tinynext_t}`,
`num_samples` ∈ `{50000, 100000, 150000, 200000}`.

#### All 4 subset sizes in parallel

```sh
curl -X POST $TRAIN_URL/train_scaling \
  -H 'Content-Type: application/json' \
  -d '{"epochs":50}'
```

Spawns 4 parallel A10G GPU jobs. Each run takes ~1–3 hours depending on subset
size. Checkpoints are saved every epoch so jobs resume automatically on
container restart.

#### Check progress

```sh
curl $TRAIN_URL/status
```

---

### Step 4 — Retrieve training logs

Pull `stats.json` (per-epoch loss and accuracy) from the Modal volume:

```sh
# All runs at once
modal volume get tinynext-imagenet-vol checkpoints/ classification/logs/ --recursive
```

Expected layout after pulling:

```
classification/logs/tinynext_t/
  50k/stats.json
  100k/stats.json
  150k/stats.json
  200k/stats.json
```

---

### Step 5 — Evaluate baseline models

```sh
cd classification/
modal deploy eval_baselines_modal.py

EVAL_URL="https://<your-username>--tinynext-eval-baselines-endpoint.modal.run"

# Trigger all 5 evals in parallel (~15-20 min)
curl -X POST $EVAL_URL/eval

# Fetch results once complete
curl $EVAL_URL/results
```

Results are also saved to the Modal volume:

```sh
modal volume get tinynext-imagenet-vol eval_baselines.json .
```

Monitor logs:

```sh
modal app logs tinynext-eval-baselines
```

---

### Step 6 — Generate plots

```sh
pip install matplotlib scipy numpy
```

#### Loss and accuracy curves

```sh
python classification/plot_loss_curves.py
# Output: classification/loss_curves.png
```

#### Scaling curve with extrapolation to full ImageNet

```sh
python classification/plot_scaling.py
# Output: classification/scaling_curve.png
```
