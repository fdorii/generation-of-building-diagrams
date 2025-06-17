import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import io
from collections import defaultdict

# Настройка страницы
st.set_page_config(
    page_title="Stable Diffusion with LoRA",
    page_icon="🎨",
    layout="wide"
)

# Функция для генерации изображения
def generate_image(prompt, lora_path, num_inference_steps, guidance_scale):
    # Базовый пайплайн
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # Загрузка LoRA
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()

    # Генерация изображения
    with st.spinner('Генерация изображения...'):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    
    return image

# Конфигурация категорий с префиксами
CATEGORY_CONFIG = {
    "Здания": {
        "prefix": "",
        "items": {
            "адм": {"default": 0},
            "произ": {"default": 0}
        }
    },
    "Резервуар вертикальный стальной": {
        "prefix": "РВС",
        "items": {
            "10к м": {"default": 0},
            "5к м": {"default": 0},
            "4.500 м": {"default": 0},
            "3к м": {"default": 0},
            "2к м": {"default": 0},
            "1к м": {"default": 0},
            "500 м": {"default": 0},
            "100 м": {"default": 0},
            "50 м": {"default": 0}
        }
    },
    "Резервуар горизонтальный стальной": {
        "prefix": "РГС",
        "items": {
            "75 м": {"default": 0},
            "60 м": {"default": 0}
        }
    },
    "Подходы": {
        "prefix": "",
        "items": {
            "ж/д": {"default": 0},
            "порт": {"default": 0}
        }
    }
}

# Основное содержимое
st.title("Генератор изображений с Stable Diffusion + LoRA")
st.write("Настройте параметры для генерации изображения")

# Создаем колонки для интерфейса
col1, col2 = st.columns([3, 2])

with col1:
    # Сбор выбранных объектов
    selected_objects = defaultdict(dict)
    
    for category, config in CATEGORY_CONFIG.items():
        # Заголовок с префиксом если есть
        header = f"{category} ({config['prefix']})" if config['prefix'] else category
        st.subheader(header)
        
        # Создаем колонки для элементов
        cols = st.columns(3)
        col_idx = 0
        
        for item, item_config in config["items"].items():
            with cols[col_idx]:
                count = st.number_input(
                    f"{item}",
                    min_value=0,
                    max_value=10,
                    value=item_config["default"],
                    step=1,
                    key=f"{category}_{item}"
                )
                if count > 0:
                    selected_objects[category][item] = count
            
            col_idx = (col_idx + 1) % 3

with col2:
    st.subheader("Параметры генерации")
    
    # Фиксированный путь к LoRA
    lora_path = "./final_lora"
    st.info(f"Используется модель: {lora_path.split('/')[-1]}")
    
    # Параметры генерации
    num_inference_steps = st.slider(
        "Количество шагов генерации",
        min_value=20,
        max_value=300,
        value=200,
        step=10,
        help="Чем больше шагов, тем детализированнее результат, но дольше генерация. Лучший вариант 50-200 шагов"
    )
    
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="""
        Контролирует насколько строго модель следует промпту:
        - 1.0: Полная свобода творчества
        - 3.0-5.0: Баланс креатива и соответствия
        - 7.0-10.0: Строгое следование промпту
        - 15.0+: Жёсткий контроль, может снизить качество
        """
    )

# Формируем итоговый промпт с префиксами
prompt_parts = []
for category, items in selected_objects.items():
    prefix = CATEGORY_CONFIG[category]["prefix"]
    for item, count in items.items():
        if count > 0:
            if prefix:
                prompt_part = f"{count} {prefix} {item}"
            else:
                prompt_part = f"{count} {item}"
            prompt_parts.append(prompt_part)

if prompt_parts:
    objects_str = ", ".join(prompt_parts)
    final_prompt = f"Схема промышленного объекта содержащего: {objects_str}."
    
    st.subheader("Итоговый промпт")
    st.code(final_prompt, language="text")

    # Кнопка генерации
    if st.button("Сгенерировать изображение", type="primary"):
        try:
            image = generate_image(
                prompt=objects_str,
                lora_path=lora_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Показ изображения
            st.image(image, caption="Сгенерированное изображение", use_container_width=True)
            
            # Кнопка скачивания
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Скачать изображение",
                data=byte_im,
                file_name="industrial_generated.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
else:
    st.warning("Пожалуйста, выберите хотя бы один объект (установите количество > 0)")