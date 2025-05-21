import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

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

        tokens = self.tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

        return {
            "pixel_values": image,
            "conditioning_image": canny,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }

# === Настройки ===
pretrained_model = "runwayml/stable-diffusion-v1-5"
controlnet_model = "lllyasviel/sd-controlnet-canny"
output_dir = "./output"
image_dir = "./data/images"
canny_dir = "./data/canny"
caption_file = "./captions.json"
resolution = 512
batch_size = 1
epochs = 50
lr = 1e-5

accelerator = Accelerator()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


# Загрузка моделей
controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model,
    controlnet=controlnet,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
).to(accelerator.device)

# Настройка LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out"],  # ключевые модули UNet
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.UNET
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
print(pipe.unet.print_trainable_parameters())


# Датасет
dataset = BuildingDataset(image_dir, canny_dir, caption_file, tokenizer, resolution)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Оптимизатор
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

pipe.unet.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        with accelerator.accumulate(pipe.unet):
            prompt_embeds = pipe._encode_prompt(
                batch["input_ids"].to(accelerator.device),
                batch["attention_mask"].to(accelerator.device),
                do_classifier_free_guidance=False
            )

            outputs = pipe.unet(
                sample=batch["pixel_values"].to(accelerator.device),
                timestep=torch.tensor([0]).to(accelerator.device),
                encoder_hidden_states=prompt_embeds,
                control=batch["conditioning_image"].to(accelerator.device),
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

pipe.unet.save_pretrained(output_dir)
print("✅ Обучено с LoRA и сохранено в:", output_dir)
