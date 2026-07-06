import os
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from ultralytics import YOLO

#1 настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#2 словарь названий классов
class_names = {
    0: "Ограничение скорости (20 км/ч)",
    1: "Ограничение скорости (30 км/ч)",
    2: "Обгон грузовым автомобилям запрещен",
    3: "Перекресток со второстепенной дорогой",
    4: "Главная дорога",
    5: "Уступите дорогу",
    6: "Движение без остановки запрещено (STOP)",
    7: "Движение запрещено",
    8: "Въезд запрещен (Кирпич)",
    9: "Препятствие",
    10: "Опасность",
    11: "Опасный поворот налево",
    12: "Ограничение скорости (50 км/ч)",
    13: "Опасный поворот направо",
    14: "Иллюзия (извилистая дорога)",
    15: "Неровная дорога",
    16: "Скользкая дорога",
    17: "Сужение дороги справа",
    18: "Дорожные работы",
    19: "Светофорное регулирование",
    20: "Пешеходный переход",
    21: "Дети",
    22: "Велосипедная дорожка",
    23: "Ограничение скорости (60 км/ч)",
    24: "Гололед/снег",
    25: "Дикие животные",
    26: "Конец всех ограничений",
    27: "Движение направо",
    28: "Движение налево",
    29: "Движение прямо",
    30: "Движение прямо или направо",
    31: "Движение прямо или налево",
    32: "Объезд препятствия справа",
    33: "Объезд препятствия слева",
    34: "Ограничение скорости (70 км/ч)",
    35: "Круговое движение",
    36: "Конец зоны запрещения обгона",
    37: "Конец зоны запрещения обгона грузовикам",
    38: "Ограничение скорости (80 км/ч)",
    39: "Конец зоны ограничения скорости (80 км/ч)",
    40: "Ограничение скорости (100 км/ч)",
    41: "Ограничение скорости (120 км/ч)",
    42: "Обгон запрещен" 
}

#3 инициализация папок и моделей
yolo_path = os.path.join("models", "yolo_traffic_fixed.pt")
classifier_path = os.path.join("models", "best_model.pth")

#загрузка детектора YOLO
detection_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None

#загрузка MobileNetV3
best_model = models.mobilenet_v3_small(weights=None)
best_model.classifier[3] = torch.nn.Linear(best_model.classifier[3].in_features, len(class_names))

if os.path.exists(classifier_path):
    best_model.load_state_dict(torch.load(classifier_path, map_location=device))
    print("Веса MobileNetV3 успешно загружены")
else:
    print(f"Файл весов {classifier_path} не найден")

best_model.to(device)
best_model.eval()

#4 предобработка
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_and_crop(image):
    """ Находит знак на фото с помощью YOLO и вырезает его """
    if detection_model is None:
        return image, "Детектор не загружен, классификация полного фото"

    results = detection_model(image)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return image, "Знаки не обнаружены, классификация полного фото"

    best_box = max(boxes, key=lambda x: x.conf[0].item())
    xyxy = best_box.xyxy[0].tolist()
    cropped_img = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    return cropped_img, f"Обнаружен знак с уверенностью {best_box.conf[0].item():.2f}"

def predict_image(img):
    if img is None:
        return None, "Пожалуйста, загрузите изображение"

    img = img.convert("RGB")
    cropped_image, status = detect_and_crop(img)

    tensor_img = predict_transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = best_model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]

    values, indices = torch.topk(probabilities, k=5)
    results_output = {class_names.get(idx.item(), f"Класс {idx.item()}"): float(val) for val, idx in zip(values, indices)}

    return cropped_image, results_output

#5 интерфейс Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Входное фото"),
    outputs=[
        gr.Image(type="pil", label="Вырезанный дорожный знак"),
        gr.Label(num_top_classes=5, label="Результат распознавания")
    ],
    title="Система распознавания дорожных знаков",
    # ИЗМЕНЕНО: описание изменено под локальную папку
    description=f"Запущено на устройстве: {device}. Модель: MobileNetV3. Веса локально подгружаются из папки models/."
)

if __name__ == "__main__":
    iface.launch(share=True)
