#!/usr/bin/env python3
"""Run the paper HEMIT evaluation protocol on 1024x1024 test tiles.

Each released test tile is evaluated in 512x512 windows with a 256-pixel
stride. Predictions are saved per window; no image data is bundled here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm.auto import tqdm

from diffvs.infer_diffusion_ft import decode_latents, encode_latents
from diffvs.modeling import MarkerTokenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="HEMIT root containing test/input and test/label.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument(
        "--split-ids",
        type=Path,
        default=REPO_ROOT / "splits" / "hemit" / "hemit_test_no_overlap_no_empty_ids.txt",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-panels", action="store_true")
    return parser.parse_args()


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def generate(
    source: torch.Tensor,
    marker_id: torch.Tensor,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    marker_encoder: MarkerTokenEncoder,
    eta: float,
) -> torch.Tensor:
    source_latents = encode_latents(vae, source)
    latents = torch.randn_like(source_latents)
    context = marker_encoder(marker_id)
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents, source_latents], dim=1)
        noise_pred = unet(model_input, timestep, encoder_hidden_states=context, return_dict=False)[0]
        latents = scheduler.step(noise_pred, timestep, latents, eta=eta).prev_sample
    return decode_latents(vae, latents)


def save_panel(source: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    width, height = source.shape[-1], source.shape[-2]
    panel = Image.new("RGB", (width * 3, height))
    panel.paste(to_pil_image(source.cpu()), (0, 0))
    panel.paste(to_pil_image(target.cpu()), (width, 0))
    panel.paste(to_pil_image(pred.cpu()), (width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main() -> None:
    args = parse_args()
    if args.window_size != 512 or args.stride != 256:
        raise ValueError("The released paper protocol uses --window-size 512 and --stride 256.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    prediction_dir = output_dir / "predictions"
    panel_dir = output_dir / "panels"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    marker_ckpt = torch.load(args.checkpoint_dir / "marker_encoder.pt", map_location="cpu")
    marker_names = list(marker_ckpt.get("markers", ["HEMIT"]))
    if marker_names != ["HEMIT"]:
        raise ValueError(f"Expected a HEMIT checkpoint with marker ['HEMIT'], got {marker_names}")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(args.checkpoint_dir / "unet").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    marker_encoder = MarkerTokenEncoder(marker_names, int(marker_ckpt["cross_attention_dim"]))
    marker_encoder.load_state_dict(marker_ckpt["state_dict"])
    marker_encoder.to(device).eval()
    marker_id = torch.zeros(1, dtype=torch.long, device=device)

    records = []
    with torch.no_grad():
        for sample_id in tqdm(load_ids(args.split_ids), desc="HEMIT sliding-window inference"):
            source_image = Image.open(args.dataset_root / "test" / "input" / sample_id).convert("RGB")
            target_image = Image.open(args.dataset_root / "test" / "label" / sample_id).convert("RGB")
            if source_image.size != target_image.size:
                raise ValueError(f"Mismatched source/target size for {sample_id}")
            width, height = source_image.size
            if width < args.window_size or height < args.window_size:
                raise ValueError(f"Tile is smaller than {args.window_size}x{args.window_size}: {sample_id}")
            stem = Path(sample_id).stem
            for y in range(0, height - args.window_size + 1, args.stride):
                for x in range(0, width - args.window_size + 1, args.stride):
                    source = to_tensor(source_image.crop((x, y, x + args.window_size, y + args.window_size))).unsqueeze(0).to(device)
                    target = to_tensor(target_image.crop((x, y, x + args.window_size, y + args.window_size)))
                    pred = generate(source, marker_id, vae, unet, scheduler, marker_encoder, args.eta)[0]
                    suffix = f"{stem}_y{y:04d}_x{x:04d}"
                    pred_path = prediction_dir / f"{suffix}.png"
                    to_pil_image(pred.cpu()).save(pred_path)
                    panel_path = None
                    if not args.skip_panels:
                        panel_path = panel_dir / f"{suffix}.png"
                        save_panel(source[0], target, pred, panel_path)
                    records.append({"sample_id": sample_id, "y": y, "x": x, "prediction_path": str(pred_path), "panel_path": str(panel_path) if panel_path else None})
    (output_dir / "inference_manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
