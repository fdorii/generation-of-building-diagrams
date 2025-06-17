import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import json
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import traceback

# Конфигурация
class Config:
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    image_dir = "./Attempt 4/data/images"
    prompt_json = "./Attempt 4/data/captions.json"  # Файл с промтами
    output_dir = "./lora_output_7"
    resolution = 512
    batch_size = 3
    gradient_accumulation_steps = 10
    learning_rate = 1e-3
    lr_warmup_steps = 100
    num_epochs = 400
    lora_rank = 5
    save_steps = 50

class TextImageDataset(Dataset):
    def __init__(self, image_dir, prompt_json, resolution=512):
        try:
            self.image_paths = []
            self.prompts = {}
            self.resolution = resolution
            
            # Проверка и загрузка изображений
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Директория с изображениями не найдена: {image_dir}")
            
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            self.image_paths = [
                os.path.join(image_dir, f) 
                for f in os.listdir(image_dir) 
                if f.lower().endswith(valid_extensions)
            ]
            
            if not self.image_paths:
                raise ValueError(f"В директории {image_dir} нет изображений с поддерживаемыми форматами {valid_extensions}")
            
            # Загрузка и проверка JSON с промтами
            if not os.path.exists(prompt_json):
                raise FileNotFoundError(f"Файл с промтами не найден: {prompt_json}")
            
            with open(prompt_json, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            
            # Проверка соответствия изображений и промтов
            missing_prompts = [
                os.path.basename(img_path) 
                for img_path in self.image_paths 
                if os.path.basename(img_path) not in self.prompts
            ]
            
            if missing_prompts:
                print(f"\n⚠️ Внимание: для {len(missing_prompts)} изображений нет промтов!")
                print("Примеры файлов без промтов:", missing_prompts[:5])
                
        except Exception as e:
            print("\n❌ Критическая ошибка при создании датасета:")
            print(f"Тип ошибки: {type(e).__name__}")
            print(f"Сообщение: {str(e)}")
            print("\nСтек вызовов:")
            traceback.print_exc()
            raise

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            filename = os.path.basename(img_path)
            prompt = self.prompts.get(filename, "")
            
            if not prompt:
                print(f"⚠️ Для файла {filename} используется пустой промпт")
            
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.resolution, self.resolution))
            img = np.array(img).astype(np.float32) / 127.5 - 1.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            return {"pixel_values": img, "prompt": prompt}
            
        except Exception as e:
            print(f"\n❌ Ошибка при обработке изображения {img_path}:")
            print(f"Тип ошибки: {type(e).__name__}")
            print(f"Сообщение: {str(e)}")
            raise

def setup_model(config):
    try:
        print("\n🔄 Инициализация модели...")
        pipe = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model, 
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        
        print("⚙️ Настройка LoRA...")
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=16,
            target_modules=["to_k", "to_q", "to_v", "proj_out"],
            init_lora_weights="gaussian",
        )
        
        unet = pipe.unet
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        
        pipe.vae = pipe.vae.half()
        
        print("✅ Модель успешно инициализирована")
        return pipe, unet
        
    except Exception as e:
        print("\n❌ Ошибка инициализации модели:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        print("\nСтек вызовов:")
        traceback.print_exc()
        raise

def train_lora():
    try:
        config = Config()
        os.makedirs(config.output_dir, exist_ok=True)
        
        print("\n📊 Конфигурация обучения:")
        print(f"Модель: {config.pretrained_model}")
        print(f"Размер батча: {config.batch_size}")
        print(f"Шагов аккумуляции: {config.gradient_accumulation_steps}")
        print(f"Эффективный размер батча: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Каталог вывода: {os.path.abspath(config.output_dir)}")
        
        # Инициализация
        dataset = TextImageDataset(config.image_dir, config.prompt_json, config.resolution)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: {
                'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
                'prompt': [item['prompt'] for item in batch]
            }
        )
        
        pipe, unet = setup_model(config)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
        
        total_steps = (len(dataloader) * config.num_epochs) // config.gradient_accumulation_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_steps
        )
        
        # Обучение
        global_step = 0
        actual_steps_since_save = 0
        
        print("\n🚀 Начало обучения...")
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{config.num_epochs}")
            
            for batch in progress_bar:
                try:
                    # Forward pass
                    pixel_values = batch['pixel_values'].half().to("cuda")
                    
                    text_inputs = pipe.tokenizer(
                        batch['prompt'],
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        return_tensors="pt",
                        truncation=True
                    ).to("cuda")
                    
                    with torch.no_grad():
                        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
                    
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, pipe.scheduler.config.num_train_timesteps, 
                        (latents.shape[0],), device="cuda"
                    )
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    
                    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                    
                    actual_steps_since_save += 1
                    
                    # Шаг оптимизации
                    if actual_steps_since_save % config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        progress_bar.set_postfix({
                            "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                            "шаг": global_step
                        })
                        
                        # Сохранение
                        if global_step % config.save_steps == 0:
                            save_path = os.path.join(config.output_dir, f"step_{global_step}")
                            unet.save_pretrained(save_path)
                            print(f"\n💾 Сохранение модели на шаге {global_step} в {save_path}")
                
                except Exception as e:
                    print(f"\n⚠️ Ошибка во время обучения (шаг {global_step}):")
                    print(f"Тип: {type(e).__name__}")
                    print(f"Сообщение: {str(e)}")
                    print("\nПродолжаем обучение со следующего батча...")
                    optimizer.zero_grad()  # Сброс градиентов
                    continue
        
        # Финальное сохранение
        final_path = os.path.join(config.output_dir, "final_lora")
        unet.save_pretrained(final_path)
        print(f"\n🎉 Обучение завершено! Финальная модель сохранена в {final_path}")
        
    except Exception as e:
        print("\n❌ Критическая ошибка в обучении:")
        print(f"Тип: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        print("\nСтек вызовов:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train_lora()