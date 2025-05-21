import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler
)
from transformers import CLIPTokenizer
from accelerate import Accelerator
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch.nn.functional as F

class BuildingDataset(Dataset):
    def __init__(self, image_dir, canny_dir, captions_file, tokenizer, resolution=512):
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        self.images = list(self.captions.keys())
        self.image_dir = image_dir
        self.canny_dir = canny_dir
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = self.captions[img_name]

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        canny = Image.open(os.path.join(self.canny_dir, img_name)).convert("RGB")

        image = self.transform(image)
        canny = self.transform(canny)

        return {
            "pixel_values": image,
            "conditioning_image": canny,
            "caption": caption,
        }

# === Настройки ===
pretrained_model = "runwayml/stable-diffusion-v1-5"
controlnet_model = "lllyasviel/sd-controlnet-canny"
output_dir = "./output"
image_dir = "./data/images"
canny_dir = "./data/canny"
caption_file = "./data/captions.json"
resolution = 384
batch_size = 1
epochs = 50
lr = 1e-5
save_every_n_steps = 10

accelerator = Accelerator(mixed_precision="fp16")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Загрузка моделей
controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model,
    controlnet=controlnet,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
).to(accelerator.device)

pipe.unet.enable_gradient_checkpointing()

pipe.scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

# === ✅ Применение LoRA к attention-слоям UNet ===
rank = 1
for name, module in pipe.unet.named_modules():
    if hasattr(module, "set_attn_processor"):
        try:
            module.set_attn_processor(LoRAAttnProcessor(hidden_size=module.to_q.in_features, rank=rank))
        except Exception:
            pass

print("✅ LoRA адаптеры применены к UNet через diffusers.")

# === Обучение ===
dataset = BuildingDataset(image_dir, canny_dir, caption_file, tokenizer, resolution)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

vae = pipe.vae
scheduler = pipe.scheduler
pipe.unet.train()

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        with accelerator.accumulate(pipe.unet):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=pipe.unet.dtype)
            cond_image = batch["conditioning_image"].to(accelerator.device, dtype=pipe.unet.dtype)

            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, _ = pipe.encode_prompt(
                prompt=batch["caption"],
                device=latents.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_image,
                return_dict=False
            )

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False
            )[0]

            loss = F.mse_loss(noise_pred.float(), noise.float())
            accelerator.backward(loss)
            optimizer.step()
            del latents, noise, noisy_latents, noise_pred, loss
            torch.cuda.empty_cache()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # 💾 Сохраняем каждые N шагов
        if step % save_every_n_steps == 0 and step != 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-step-{epoch}-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            pipe.unet.save_attn_procs(checkpoint_dir)
            print(f"✅ Checkpoint сохранён: {checkpoint_dir}")

# Финальное сохранение
pipe.unet.save_attn_procs(output_dir)
print("✅ Финальные LoRA веса сохранены в:", output_dir)