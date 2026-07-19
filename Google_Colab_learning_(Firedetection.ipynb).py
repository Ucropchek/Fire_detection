'''
Код ноутбука Google Colab
'''

# 1
!pip install roboflow ultralytics -q
# 2
!pip install "sympy>=1.13.3,<1.14" -q
# 3
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = "/content/drive/MyDrive/fire_detection_runs"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Результаты будут сохраняться в: {SAVE_DIR}")
# 4
from roboflow import Roboflow

rf = Roboflow(api_key="FgTggNleBDSyiYJD9oeL")
project = rf.workspace("georges-workspace-fnhr8").project("fire_detection-dtyvi")

print("Доступные версии:")
for v in project.versions():
    print(" -", v.version)

dataset = project.version(3).download("yolov11")
print(f"\nДатасет скачан: {dataset.location}")
# 5
import glob
from collections import Counter

DATASET_PATH = dataset.location

label_files = glob.glob(f"{DATASET_PATH}/**/*.txt", recursive=True)
label_files = [f for f in label_files if "data.yaml" not in f]

class_counts = Counter()
for label_file in label_files:
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                class_counts[int(parts[0])] += 1
            except (ValueError, IndexError):
                continue

print(f"Найдено файлов разметки: {len(label_files)}")
print("Распределение классов:")
for class_id, count in sorted(class_counts.items()):
    print(f"  Класс {class_id}: {count} инстансов")
# 6
import glob
import yaml
from collections import Counter

with open(f"{dataset.location}/data.yaml") as f:
    data_yaml = yaml.safe_load(f)

print("Классы в data.yaml:", data_yaml["names"])

label_files = glob.glob(f"{dataset.location}/**/*.txt", recursive=True)
label_files = [f for f in label_files if "data.yaml" not in f]

class_counts = Counter()
for label_file in label_files:
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    class_counts[int(line.split()[0])] += 1
                except (ValueError, IndexError):
                    continue

print("\nФинальное распределение классов:")
for class_id, count in sorted(class_counts.items()):
    class_name = data_yaml["names"][class_id]
    print(f"  Класс {class_id} ('{class_name}'): {count} инстансов")
# 7
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    patience=30,
    device=0,
    project="fire_detection_runs",
    name="yolo11s_fire",

    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    dropout=0.1,
    weight_decay=0.0005,
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,

    val=True,
    save_period=10,
)

print("Обучение завершено!")
# 8
import shutil, os

local_path = "/content/runs/detect/fire_detection_runs/yolo11s_fire"
drive_path = f"{SAVE_DIR}/yolo11s_fire"

if os.path.exists(drive_path):
    shutil.rmtree(drive_path)

shutil.copytree(local_path, drive_path)
print(f"✅ Сохранено: {drive_path}")
# 9
best_model = YOLO(f"{drive_path}/weights/best.pt")
metrics = best_model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
# 10
import os
import glob
from IPython.display import Image, display

print("Файлы в drive_path:")
for f in sorted(os.listdir(drive_path)):
    print(" -", f)

pr_files = glob.glob(f"{drive_path}/*PR_curve*") + glob.glob(f"{drive_path}/*pr_curve*")
print("\nНайденные PR curve файлы:", pr_files)

results_png = f"{drive_path}/results.png"
confusion_png = f"{drive_path}/confusion_matrix.png"

if os.path.exists(results_png):
    print("\nГрафик обучения (loss, mAP по эпохам):")
    display(Image(results_png))

if os.path.exists(confusion_png):
    print("\nМатрица ошибок:")
    display(Image(confusion_png))

if pr_files:
    print("\nPrecision-Recall кривая:")
    display(Image(pr_files[0]))
else:
    print("\n⚠️ PR curve файл не найден в drive_path")

print("\nПапки в /content/runs/detect/:")
if os.path.exists("/content/runs/detect/"):
    for f in sorted(os.listdir("/content/runs/detect/")):
        print(" -", f)
# 11
import pandas as pd

df = pd.read_csv(f"{drive_path}/results.csv")
print("Последние 10 эпох:")
print(df.tail(10))