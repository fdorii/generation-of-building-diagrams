import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import tempfile
import os
import sys
from finish.api import generate_image, CATEGORY_CONFIG

# Добавляем путь к модулю приложения в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImageGeneration(unittest.TestCase):
    """Тестирование генерации изображений"""
    
    def setUp(self):
        # Создаем временную папку для тестовой LoRA модели
        self.temp_dir = tempfile.mkdtemp()
        self.lora_path = os.path.join(self.temp_dir, "test_lora")
        os.makedirs(self.lora_path, exist_ok=True)
        
        # Тестовые параметры
        self.test_prompt = "1 РВС 10к м, 2 здания адм"
        self.num_steps = 50
        self.guidance_scale = 7.5

        # Создаем тестовое изображение
        self.test_image = Image.new('RGB', (512, 512), color='red')

    def tearDown(self):
        # Очистка временных файлов
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_generate_image_success(self, mock_peft, mock_pipeline):
        """Тест успешной генерации изображения"""
        # Настраиваем моки
        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.images = [self.test_image]
        mock_pipeline.return_value = mock_pipe
        
        # Мокируем peft
        mock_unet = MagicMock()
        mock_unet.merge_and_unload.return_value = mock_unet
        mock_peft.return_value = mock_unet
        
        # Вызываем функцию
        result = generate_image(
            prompt=self.test_prompt,
            lora_path=self.lora_path,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale
        )

        # Проверяем результаты
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (512, 512))
        mock_pipeline.assert_called_once_with(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        mock_peft.assert_called_once()

    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_generate_image_failure(self, mock_pipeline):
        """Тест обработки ошибок генерации"""
        # Настраиваем mock для вызова исключения
        mock_pipeline.side_effect = RuntimeError("CUDA out of memory")

        # Импортируем функцию после мокирования

        with self.assertRaises(RuntimeError):
            generate_image(
                prompt=self.test_prompt,
                lora_path=self.lora_path,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale
            )

class TestPromptGeneration(unittest.TestCase):
    """Тестирование формирования промптов"""
    
    def test_prompt_construction(self):
        """Тест построения промпта из выбранных объектов"""
        # Импортируем конфигурацию и функцию после мокирования
        
        test_cases = [
            # Пустой ввод
            ({}, ""),
            # Один объект с префиксом
            ({"Резервуар вертикальный стальной": {"10к м": 1}}, "1 РВС 10к м"),
            # Один объект без префикса
            ({"Здания": {"адм": 2}}, "2 адм"),
            # Несколько объектов
            ({
                "Резервуар вертикальный стальной": {"10к м": 1},
                "Подходы": {"ж/д": 3}
            }, "1 РВС 10к м, 3 ж/д"),
            # Объекты из всех категорий
            ({
                "Здания": {"адм": 1},
                "Резервуар вертикальный стальной": {"5к м": 2},
                "Резервуар горизонтальный стальной": {"75 м": 1},
                "Подходы": {"порт": 1}
            }, "1 адм, 2 РВС 5к м, 1 РГС 75 м, 1 порт")
        ]
        
        for input_data, expected in test_cases:
            with self.subTest(input_data=input_data, expected=expected):
                prompt_parts = []
                for category, items in input_data.items():
                    prefix = CATEGORY_CONFIG[category]["prefix"]
                    for item, count in items.items():
                        if count > 0:
                            if prefix:
                                prompt_part = f"{count} {prefix} {item}"
                            else:
                                prompt_part = f"{count} {item}"
                            prompt_parts.append(prompt_part)
                
                result = ", ".join(prompt_parts) if prompt_parts else ""
                self.assertEqual(result, expected)

class TestCategoryConfig(unittest.TestCase):
    """Тестирование конфигурации категорий"""
    
    def test_category_structure(self):
        """Проверка структуры конфигурации категорий"""
        
        required_keys = ["prefix", "items"]
        
        for category, config in CATEGORY_CONFIG.items():
            with self.subTest(category=category):
                # Проверяем наличие обязательных ключей
                for key in required_keys:
                    self.assertIn(key, config)
                
                # Проверяем структуру items
                self.assertIsInstance(config["items"], dict)
                for item, item_config in config["items"].items():
                    self.assertIn("default", item_config)

if __name__ == '__main__':
    unittest.main()