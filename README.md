---

Этот репозиторий представляет собой диплом на тему `Разработка интерфейса и обучение модели генерации строительных схем размещения резервуарного парка для хранения битума` со всеми пробами и ошибками: 

---

## **Структура работы**

<details>
  <summary><h2>⚙️ Структура данных </h2></summary>
  <ul>
  	    <li><b>info_data.ipynb</b> - Информация по классам</li>
        <li><b>examples_of_source_images</b> - примеры исходных изображений <code>до</code> преобразования</li>
        <li><b>last_data</b> - изображения и разметка после <code>первого</code> преобразования</li>
	      <li><b>new_data</b> - изображения и разметка после <code>второго</code> преобразования, было добавлено больше цвета к ихображениям, а также новые теги</li>
	</ul>
</details>

<details>
  <summary><h2>1️⃣ Попытка 1: Обучение LoRa на Flux</h2></summary>
  Для обучения закрытой диффузионной модели Flux использовалось приложение Pinokio (<a href="https://boosty.to/nevskiyart/posts/c37401ab-a5a4-4495-a4f7-d5c272e6433f">ссылка на гайд обучения</a>). В целом обучение прошло нормально, но из-за ограниченных ресурсов для обучения не хватило времени, данных для более корректного обучения.
  <br>
  <br>
  <b>test_lora.ipynb</b> - Использование уже обученной модели.
  <br>
  <b>courseworkmodel.safetensors</b> - веса обученной модели.
</details>

<details>
  <summary><h2>2️⃣ Попытка 2: Обучение UNet2DModel</h2></summary>
  (<a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=r5PM6vOQPISl">ссылка на гайд обучения</a>). В целом обучение прошло нормально, но из-за ограниченных ресурсов для обучения не хватило времени, данных для более корректного обучения.
  <br>
  <br>
  <b>test_training.ipynb</b> - Обученние модели.
  <br>
  <b>runs</b> - Логи обучения.
</details>

<details>
  <summary><h2>3️⃣ Попытка 3: Обучение UNet2DModel на новых данных</h2></summary>
  Были изменены исходные фотографии во второй раз, добавлено два тега, увеличилось количество эпох, но лучше не стало...
  (<a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=r5PM6vOQPISl">ссылка на гайд обучения</a>). В целом обучение прошло нормально, но из-за ограниченных ресурсов для обучения не хватило времени, данных для более корректного обучения.
  <br>
  <br>
  <b>test_training_new_data.ipynb</b> - Обученние модели.
  <!-- <br>
  <b>runs</b> - Логи обучения. -->
</details>