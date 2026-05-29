# VL-JEPA Minimal Reproduction

This is a clean minimal reproduction scaffold for VL-JEPA-style training.
It keeps only the pieces needed for:

```text
video -> frozen V-JEPA-style encoder -> predictor -> predicted text embedding
caption -> frozen Y-encoder -> target text embedding
loss = 1 - cosine(predicted, target)
```

The existing toy repository is intentionally left untouched.

## Layout

```text
vl-jepa-repro/
  configs/default.yaml
  src/
    data/                 # video-text manifest dataset
    vjepa_backbone/       # minimal V-JEPA-style 3D ViT, checkpoint loader, transforms
    vl_jepa/              # Y-encoder, predictor, model wrapper, loss
  tests/
  train_vl_jepa.py
```

## Manifest Format

Use a CSV file with exactly these columns:

```csv
video_path,caption,split
/path/to/video.mp4,a person is cooking,train
/path/to/video2.mp4,a dog runs on grass,val
```

## MSR-VTT Preparation

If you downloaded `friedrichor/MSR-VTT` from Hugging Face into `data/msr-vtt`,
convert its JSON metadata into this repo's CSV manifest format:

```bash
python scripts/prepare_msrvtt_manifest.py
```

By default this reads:

```text
data/msr-vtt/msrvtt_train_7k.json
data/msr-vtt/msrvtt_test_1k.json
```

and writes:

```text
data/train_manifest.csv
data/val_manifest.csv
```

MSR-VTT train JSON stores one video with a list of captions, while this training
code expects one `video_path,caption,split` row per video-caption pair. The
script expands those caption lists into separate training rows.

## Quick Smoke Test

From this folder:

```bash
pytest
```

The tests use a tiny random backbone and a dummy frozen text encoder, so they do not
require downloading V-JEPA checkpoints or EmbeddingGemma.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Edit `configs/default.yaml`, then run:

```bash
python train_vl_jepa.py --config configs/default.yaml
```

The default config points at `google/embeddinggemma-300m` for the frozen
Y-encoder. You must download or point to a V-JEPA checkpoint yourself via
`model.vjepa_checkpoint`.

## AutoDL

On AutoDL, keep code in `/root/autodl-tmp/vl-jepa-repro` and keep large assets
outside Git:

```text
/root/autodl-tmp/models/vitl16.pth.tar
/root/autodl-tmp/models/embeddinggemma-300m
/root/autodl-tmp/data/msr-vtt
```

Clone or update the code:

```bash
cd /root/autodl-tmp
git clone https://github.com/jsw24000/vl-jepa-repro.git
cd vl-jepa-repro
```

If the repo already exists on AutoDL:

```bash
cd /root/autodl-tmp/vl-jepa-repro
git pull
```

Prepare the remote config:

```bash
cp configs/autodl_24gb.example.yaml configs/autodl_24gb.yaml
```

After downloading MSR-VTT on AutoDL, build manifests with:

```bash
python scripts/prepare_msrvtt_manifest.py \
  --dataset-root /root/autodl-tmp/data/msr-vtt \
  --videos-root /root/autodl-tmp/data/msr-vtt/videos/video \
  --train-out /root/autodl-tmp/vl-jepa-repro/data/train_manifest.csv \
  --val-out /root/autodl-tmp/vl-jepa-repro/data/val_manifest.csv \
  --strict-videos
```

Run training:

```bash
python train_vl_jepa.py --config configs/autodl_24gb.yaml
```

Run text-to-video retrieval with a trained checkpoint:

```bash
python scripts/retrieve_videos.py \
  --config configs/autodl_24gb.yaml \
  --checkpoint outputs/vl_jepa_epoch_1.pt \
  --query "a person is cooking" \
  --top-k 5 \
  --cache outputs/val_video_embeddings.pt
```

The first run builds the video embedding cache from the validation manifest.
Later runs with the same `--cache` reuse those video embeddings and only encode
the new text query.

## Attribution

The backbone follows the public V-JEPA design: a video Vision Transformer with
3D tubelet patch embedding and fixed sinusoidal positional embeddings. It is a
small, local implementation intended for VL-JEPA experiments, not a copy of the
full V-JEPA training/evaluation repository.

Reference: https://github.com/facebookresearch/jepa
