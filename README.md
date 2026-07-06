# Traffic Sign Recognition

## Описание проекта

Проект выполнен в рамках учебной практики.

Цель работы: разработка модели распознавания дорожных знаков на основе методов глубокого обучения и сравнение нескольких современных архитектур нейронных сетей.

В ходе работы были исследованы различные модели компьютерного зрения, проведено обучение, выполнена оценка качества и разработан прототип приложения для распознавания дорожных знаков.

---

## Используемый датасет

German Traffic Sign Recognition Benchmark (GTSRB)

Количество изображений: **73 139**

Количество классов: **43**

Источник:
https://www.kaggle.com/datasets/flo2607/traffic-signs-classification

---

## Использованные архитектуры

- ResNet50
- DenseNet121
- MobileNetV3-Large
- EfficientNet-B0
- Vision Transformer (ViT-B16)

Для локализации дорожных знаков использовалась модель YOLO.

---

## Лучшие результаты

| Модель | Accuracy |
|---------|----------|
| ResNet50 | 99.43 % |
| ViT-B16 | 99.22 % |
| MobileNetV3 | 98.85 % |
| EfficientNet-B0 | 98.47 % |
| DenseNet121 | 97.99 % |

---

## Структура проекта

```

models/

plots/

results/

notebooks/

app/

README.md

requirements.txt

```

---

## Используемые библиотеки

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Ultralytics YOLO
- Gradio

---

## Запуск проекта

1. Установить зависимости

```

pip install -r requirements.txt

```

2. Открыть файл

```

train.ipynb

```

3. Выполнить все ячейки.
