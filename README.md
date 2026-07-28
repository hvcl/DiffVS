# Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model

**AAAI 2026 Accepted**

This repository provides the implementation of:

> **Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model**  
> Hyun-Jic Oh, Junsik Kim, Zhiyi Shi, Yichen Wu, Yu-An Chen, Peter K. Sorger, Hanspeter Pfister, Won-Ki Jeong

The code trains a marker-wise conditioned Diffusion-FT model for virtual staining from histology images. The current release includes training code for the ORION-CRC multiplex IF setting and the HEMIT paired translation setting.

## Overview

![Overview Figure](Figure/overview.png)

DiffVS follows a two-stage training recipe. Stage 1 performs Marigold-style paired latent diffusion training: the denoiser receives the noisy target-stain latent concatenated with the source-image latent, while a learnable marker token specifies the requested marker or stain. Stage 2 applies Diffusion-FT to the Stage-1 checkpoint for efficient one-step virtual staining.

## Repository Structure

```text
DiffVS/
  Figure/
    overview.png
  configs/
    orion_markers.txt
scripts/
  generate_hemit_test_split.py       # reproduce the paper HEMIT test-ID selection
  infer_hemit_sliding_window.py      # paper-protocol HEMIT sliding-window inference
  train_orion_stage1_marigold.sh
    train_orion_stage2_diffusion_ft.sh
    train_hemit_stage1_marigold.sh
    train_hemit_stage2_diffusion_ft.sh
    infer_orion_diffusion_ft.sh
    infer_hemit_diffusion_ft.sh
  src/diffvs/
    datasets.py
    modeling.py
    train_stage1_marigold.py
    train_stage2_diffusion_ft.py
    infer_diffusion_ft.py
  requirements.txt
```

## Installation

```bash
git clone https://github.com/hvcl/DiffVS.git
cd DiffVS

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The training scripts use Hugging Face models. Make sure you have access to the corresponding model repositories and are logged in if required:

```bash
huggingface-cli login
```

## Data Layout

All paths below are examples. Replace them with your local dataset locations.

### ORION-CRC

Expected layout:

```text
/path/to/ORIONCRC_dataset_tile_20x/
  train_dataframe.csv
  val_dataframe.csv
  test_dataframe.csv
  he/
  if/
```

The dataframe files must contain at least:

```text
image_path,target_path
```

`image_path` points to the H&E tile and `target_path` points to the multiplex IF TIFF tile. Relative paths are resolved under `DATASET_ROOT`.

### HEMIT

Expected layout:

```text
/path/to/HEMIT/
  train/
    input/
    label/
  val/
    input/
    label/
  test/
    input/
    label/
```

The HEMIT loader treats the dataset as a single target-domain virtual staining task and uses one learnable target token named `HEMIT`.

### HEMIT preprocessing and paper-test split

The original HEMIT input/target pairs are 1024x1024 tiles extracted from WSIs
with a 512-pixel stride. For training, we sampled a paired random 512x512 crop
from each 1024x1024 tile at every iteration, with synchronized horizontal flip
and 90-degree rotation augmentation. The released HEMIT training loader and
both HEMIT training launchers implement this protocol explicitly; the tiles are
not resized before training.

For the paper test protocol, we first removed overlap at the 1024x1024-tile
level by retaining tiles whose two `_patch_<row>_<column>` indices were both
even. We then excluded six empty tiles. The resulting 292 sample IDs, the six
excluded IDs, and a script that reproduces the selection are provided in
[`splits/hemit/`](splits/hemit/README.md). The dataset images are not included;
they remain subject to the HEMIT dataset access terms.

At inference, each selected 1024x1024 tile is processed as 512x512 windows
with a 256-pixel stride. Thus the model's per-forward input size is 512x512 in
both training and inference. Use the paper-protocol implementation as follows:

```bash
python scripts/infer_hemit_sliding_window.py \
  --dataset-root /path/to/HEMIT \
  --checkpoint-dir /path/to/stage2-checkpoint \
  --output-dir ./outputs/hemit_paper_inference
```

## Training

The released code mirrors the two-stage procedure in the paper:

1. **Stage 1: Marigold-style training.** Train a paired latent diffusion model conditioned on the source image latent and the marker token.
2. **Stage 2: Diffusion-FT.** Initialize from the Stage-1 checkpoint, use the fixed one-step denoising timestep, reconstruct the clean latent prediction, decode it with the frozen VAE, and optimize the image-space loss `0.5 * L1 + 0.5 * L2` against the target image.

### ORION-CRC

Stage 1:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
AUGMENTED_DIR=/path/to/ORIONCRC_tile_20x_he_norm \
OUTPUT_DIR=./outputs/orion_stage1_marigold \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=15 \
bash scripts/train_orion_stage1_marigold.sh
```

Stage 2:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
AUGMENTED_DIR=/path/to/ORIONCRC_tile_20x_he_norm \
STAGE1_CHECKPOINT_DIR=./outputs/orion_stage1_marigold/stage1-checkpoint-epoch-15 \
OUTPUT_DIR=./outputs/orion_stage2_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=5 \
bash scripts/train_orion_stage2_diffusion_ft.sh
```

To train a subset of markers:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
OUTPUT_DIR=./outputs/orion_stage1_panck_sma \
bash scripts/train_orion_stage1_marigold.sh \
  --markers Hoechst Pan-CK SMA
```

### HEMIT

Stage 1:

```bash
DATASET_ROOT=/path/to/HEMIT \
OUTPUT_DIR=./outputs/hemit_stage1_marigold \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=100 \
bash scripts/train_hemit_stage1_marigold.sh
```

Stage 2:

```bash
DATASET_ROOT=/path/to/HEMIT \
STAGE1_CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
OUTPUT_DIR=./outputs/hemit_stage2_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=5 \
bash scripts/train_hemit_stage2_diffusion_ft.sh
```

## Inference

### ORION-CRC

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
CHECKPOINT_DIR=./outputs/orion_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
OUTPUT_DIR=./outputs/orion_inference \
bash scripts/infer_orion_diffusion_ft.sh
```

### HEMIT

```bash
DATASET_ROOT=/path/to/HEMIT \
CHECKPOINT_DIR=./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
OUTPUT_DIR=./outputs/hemit_inference \
bash scripts/infer_hemit_diffusion_ft.sh
```

Inference writes generated images, three-column panels, and an `inference_manifest.json`.

## Checkpoints

Training writes checkpoints to `OUTPUT_DIR`:

```text
stage1-checkpoint-epoch-*/
stage2-checkpoint-epoch-*/
config.json
logs/
```

Each checkpoint stores:

```text
unet/
marker_encoder.pt
markers
```

Large checkpoints and generated outputs are ignored by git.

## Citation

If this code or paper is useful for your research, please cite:

```bibtex
@inproceedings{oh2026virtual,
  title={Virtual multiplex staining for histological images using a marker-wise conditioned diffusion model},
  author={Oh, Hyun-Jic and Kim, Junsik and Shi, Zhiyi and Wu, Yichen and Chen, Yu-An and Sorger, Peter K and Pfister, Hanspeter and Jeong, Won-Ki},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={10},
  pages={8168--8176},
  year={2026}
}
```

## Acknowledgements

This implementation builds on ideas and components from several excellent open-source projects. In particular, we thank the authors of:

- [Marigold](https://github.com/prs-eth/Marigold)
- [diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)

Please also cite those projects when using their models, code, or training recipes.

## License

Please check the licenses of this repository and the referenced pretrained models before commercial use. Dataset access and redistribution are governed by the original dataset providers.
