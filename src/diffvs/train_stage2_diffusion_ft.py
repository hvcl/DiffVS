from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .datasets import DEFAULT_ORION_MARKERS, build_dataset
    from .modeling import MarkerTokenEncoder, expand_unet_input_channels
except ImportError:
    from datasets import DEFAULT_ORION_MARKERS, build_dataset
    from modeling import MarkerTokenEncoder, expand_unet_input_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Diffusion-FT from a Stage-1 Marigold-style checkpoint."
    )
    parser.add_argument("--dataset", choices=["orion", "hemit"], required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--markers", nargs="+", default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--augmented_dir", type=str, default="")
    parser.add_argument("--augmented_prob", type=float, default=0.0)
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--stage1_checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--marker_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    return parser.parse_args()


def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latents / vae.config.scaling_factor).sample
    return (images / 2.0 + 0.5).clamp(0, 1)


def predict_x0(
    scheduler: DDPMScheduler,
    noisy_latents: torch.Tensor,
    model_prediction: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Convert epsilon/v prediction to a clean latent prediction."""
    alphas = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas[timesteps].view(-1, 1, 1, 1)
    beta_t = 1.0 - alpha_t
    if scheduler.config.prediction_type == "epsilon":
        return (noisy_latents - beta_t.sqrt() * model_prediction) / alpha_t.sqrt()
    if scheduler.config.prediction_type == "v_prediction":
        return alpha_t.sqrt() * noisy_latents - beta_t.sqrt() * model_prediction
    raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    marker_ckpt_path = Path(args.stage1_checkpoint_dir) / "marker_encoder.pt"
    marker_ckpt = torch.load(marker_ckpt_path, map_location="cpu")
    marker_names = args.markers or list(marker_ckpt.get(
        "markers",
        DEFAULT_ORION_MARKERS if args.dataset == "orion" else ["HEMIT"],
    ))
    dataset, marker_names = build_dataset(
        dataset=args.dataset,
        root_dir=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        markers=marker_names,
        max_rows=args.max_rows,
        augmented_dir=args.augmented_dir or None,
        augmented_prob=args.augmented_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(Path(args.stage1_checkpoint_dir) / "unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    unet = expand_unet_input_channels(unet, in_channels=unet.config.in_channels)
    marker_encoder = MarkerTokenEncoder(
        marker_names=marker_names,
        cross_attention_dim=unet.config.cross_attention_dim,
    )
    marker_encoder.load_state_dict(marker_ckpt["state_dict"])

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    vae.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": unet.parameters(), "lr": args.learning_rate},
            {"params": marker_encoder.parameters(), "lr": args.marker_learning_rate},
        ]
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=len(loader) * args.num_epochs,
    )

    unet, marker_encoder, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, marker_encoder, optimizer, loader, lr_scheduler
    )
    vae = accelerator.prepare_model(vae, evaluation_mode=True)

    config = vars(args).copy()
    config["markers"] = list(marker_names)
    config["stage"] = "stage2_diffusion_ft"
    config["model_variant"] = "marker-wise-diffusion-ft"
    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    global_step = 0
    for epoch in range(args.num_epochs):
        unet.train()
        marker_encoder.train()
        progress = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
        progress.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in loader:
            with accelerator.accumulate(unet):
                source = batch["source"].to(accelerator.device)
                target = batch["target"].to(accelerator.device)
                marker_ids = batch["marker_id"].to(accelerator.device)

                with torch.no_grad():
                    source_latents = encode_latents(vae, source)
                    target_latents = encode_latents(vae, target)

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                    dtype=torch.long,
                )
                noisy_target = noise_scheduler.add_noise(target_latents, noise, timesteps)
                model_input = torch.cat([noisy_target, source_latents], dim=1)
                marker_context = marker_encoder(marker_ids)

                model_prediction = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=marker_context,
                    return_dict=False,
                )[0]
                predicted_latents = predict_x0(
                    noise_scheduler,
                    noisy_target,
                    model_prediction,
                    timesteps,
                )
                predicted_images = decode_latents(vae, predicted_latents)
                l1_loss = F.l1_loss(predicted_images.float(), target.float(), reduction="mean")
                l2_loss = F.mse_loss(predicted_images.float(), target.float(), reduction="mean")
                loss = 0.5 * l1_loss + 0.5 * l2_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(unet.parameters()) + list(marker_encoder.parameters()), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()), lr=lr_scheduler.get_last_lr()[0])

        progress.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0):
            ckpt_dir = Path(args.output_dir) / f"stage2-checkpoint-epoch-{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "markers": list(marker_names),
                    "state_dict": accelerator.unwrap_model(marker_encoder).state_dict(),
                    "cross_attention_dim": accelerator.unwrap_model(marker_encoder).embedding.embedding_dim,
                },
                ckpt_dir / "marker_encoder.pt",
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
