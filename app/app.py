import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from ultralytics import YOLO

#1 настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#2 словарь названий классов
class_names = {
    0: "Ограничение скорости (20 км/ч)", 1: "Ограничение скорости (30 км/ч)", 
    2: "Ограничение скорости (50 км/ч)", 3: "Ограничение скорости (60 км/ч)", 
    4: "Ограничение скорости (70 км/ч)", 5: "Ограничение скорости (80 км/ч)", 
    6: "Конец зоны ограничения скорости (80 км/ч)", 7: "Ограничение скорости (100 км/ч)", 
    8: "Ограничение скорости (120 км/ч)", 9: "Обгон запрещен", 
    10: "Обгон грузовым автомобилям запрещен", 11: "Перекресток со второстепенной дорогой", 
    12: "Главная дорога", 13: "Уступите дорогу", 14: "Движение без остановки запрещено (STOP)", 
    15: "Движение запрещено", 16: "Въезд запрещен (Кирпич)", 17: "Препятствие", 
    18: "Опасность", 19: "Опасный поворот налево", 20: "Опасный поворот направо", 
    21: "Иллюзия (извилистая дорога)", 22: "Неровная дорога", 23: "Скользкая дорога", 
    24: "Сужение дороги справа", 25: "Дорожные работы", 26: "Светофорное регулирование", 
    27: "Пешеходный переход", 28: "Дети", 29: "Велосипедная дорожка", 
    30: "Гололед/снег", 31: "Дикие животные", 32: "Конец всех ограничений", 
    33: "Движение направо", 34: "Движение налево", 35: "Движение прямо", 
    36: "Движение прямо или направо", 37: "Движение прямо или налево", 38: "Объезд препятствия справа", 
    39: "Объезд препятствия слева", 40: "Круговое движение", 41: "Конец зоны запрещения обгона", 
    42: "Конец зоны запрещения обгона грузовикам"
}

#3 инициализация моделей
#загрузка детектора YOLO
yolo_path = os.path.join("models", "yolo_traffic_fixed.pt")
detection_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None

#загрузка лучшей классификационной модели
best_model = torchvision.models.resnet50(pretrained=False) 
num_ftrs = best_model.fc.in_features
best_model.fc = torch.nn.Linear(num_ftrs, len(class_names))

classifier_path = os.path.join("models", "best_model.pth")
if os.path.exists(classifier_path):
    best_model.load_state_dict(torch.load(classifier_path, map_location=device))
best_model.to(device)
best_model.eval()

#4 предобработка изображений для классификатора
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
    
    # Берем первый найденный бокс с наибольшей уверенностью
    best_box = max(boxes, key=lambda x: x.conf[0].item())
    xyxy = best_box.xyxy[0].tolist()
    cropped_img = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    return cropped_img, f"Обнаружен знак с уверенностью {best_box.conf[0].item():.2f}"

def predict_image(img):
    if img is None:
        return None, "Пожалуйста, загрузите изображение"
    
    #конвертация в RGB
    img = img.convert("RGB")
    
    #детекция и кроп
    cropped_image, status = detect_and_crop(img)
    
    #классификация
    tensor_img = predict_transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = best_model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    #топ-5 предсказаний
    values, indices = torch.topk(probabilities, k=5)
    results_output = {class_names.get(idx.item(), f"Класс {idx.item()}"): float(val) for val, idx in zip(values, indices)}
    
    return cropped_image, results_output

#5 интерфейс Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Входное фото (дорожная ситуация)"),
    outputs=[
        gr.Image(type="pil", label="Вырезанный дорожный знак"),
        gr.Label(num_top_classes=5, label="Результат распознавания")
    ],
    title="Система распознавания дорожных знаков (TSR)",
    description=f"Запущено на устройстве: {device}. Приложение локально подгружает веса из папки models/."
)

if __name__ == "__main__":
    #запускаем локальный веб-сервер
    iface.launch(server_name="127.0.0.1", server_port=7860)
