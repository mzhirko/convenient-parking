import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


# Конфигурация, которую будет использовать библиотека Mask-RCNN.
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # в датасете COCO находится 80 классов + 1 фоновый класс.
    DETECTION_MIN_CONFIDENCE = 0.6


# Фильтруем список результатов распознавания, чтобы остались только автомобили.
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # Если найденный объект не автомобиль, то пропускаем его.
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Корневая директория проекта.
ROOT_DIR = Path(".")

# Директория для сохранения логов и обученной модели.
MODEL_DIR = ROOT_DIR / "logs"

# Локальный путь к файлу с обученными весами.
COCO_MODEL_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Загружаем датасет COCO при необходимости.
if not COCO_MODEL_PATH.exists():
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Директория с изображениями для обработки.
IMAGE_DIR = ROOT_DIR / "images"

# Видеофайл или камера для обработки — вставьте значение 0, если нужно использовать камеру, а не видеофайл.
VIDEO_SOURCE = "test_images/VID_20201106_170820.mp4"

# Создаём модель Mask-RCNN в режиме вывода.
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Загружаем предобученную модель.
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Местоположение парковочных мест.
parked_car_boxes = None

# Загружаем видеофайл, для которого хотим запустить распознавание.
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Проходимся в цикле по каждому кадру.
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Конвертируем изображение из цветовой модели BGR (используется OpenCV) в RGB.
    rgb_image = frame[:, :, ::-1]

    # Подаём изображение модели Mask R-CNN для получения результата.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN предполагает, что мы распознаём объекты на множественных изображениях.
    # Мы передали только одно изображение, поэтому извлекаем только первый результат.
    r = results[0]

    # Переменная r теперь содержит результаты распознавания:
    # - r['rois'] — ограничивающая рамка для каждого распознанного объекта;
    # - r['class_ids'] — идентификатор (тип) объекта;
    # - r['scores'] — степень уверенности;
    # - r['masks'] — маски объектов (что даёт вам их контур).

    # Фильтруем результат для получения рамок автомобилей.
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    print("Cars found in frame of video:")

    # Отображаем каждую рамку на кадре.
    for box in car_boxes:
        print("Car:", box)

        y1, x1, y2, x2 = box

        # Рисуем рамку.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Показываем кадр на экране.
    cv2.imshow('Video', frame)

    # Нажмите 'q', чтобы выйти.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очищаем всё после завершения.
video_capture.release()
cv2.destroyAllWindows()