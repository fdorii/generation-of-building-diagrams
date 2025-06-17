import os
import json
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from transformers import CLIPTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model

# ========== Класс датасета ==========
class BuildingDataset(Dataset):
    def __init__(self, image_dir, canny_dir, captions_file, tokenizer, resolution=512):
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.canny_dir = canny_dir
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        
        image = Image.open(img_path).convert("RGB")
        canny = Image.open(os.path.join(self.canny_dir, img_name)).convert("RGB")
        caption = self.captions.get(img_name, "")
        
        image = self.transform(image).to(torch.float16)
        canny = self.transform(canny).to(torch.float16)

        return {
            "pixel_values": image,
            "conditioning_image": canny,
            "caption": caption,
        }

# ========== Парсинг аргументов ==========
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--canny_dir", type=str, required=True)
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

# ========== Основная функция ==========
def main():
    args = parse_args()
    
    # Инициализация Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[kwargs]
    )
    
    if args.seed is not None:
        set_seed(args.seed)

    # Загрузка моделей
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(accelerator.device)

    # Настройка LoRA
    unet = pipe.unet
    unet.requires_grad_(False)
    


    # Настройка LoRA для UNet
    lora_config = LoraConfig(
        r=4,  
        target_modules=["to_q", "to_v"], 
        lora_alpha=8,
        lora_dropout=0.1,
    )

    unet = get_peft_model(unet, lora_config)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Подготовка данных
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    dataset = BuildingDataset(
        image_dir=args.image_dir,
        canny_dir=args.canny_dir,
        captions_file=args.caption_file,
        tokenizer=tokenizer,
        resolution=args.resolution
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Подготовка к обучению с Accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Цикл обучения
    global_step = 0
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                # Прямой проход
                latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Получение эмбеддингов текста
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                # ControlNet прямой проход
                down_samples, mid_sample = pipe.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=batch["conditioning_image"],
                    return_dict=False
                )

                # Предсказание шума
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                ).sample

                # Расчет потерь
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Обратный проход
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Логирование и сохранение
            if global_step % args.save_every_n_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

    # Финальное сохранение
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()