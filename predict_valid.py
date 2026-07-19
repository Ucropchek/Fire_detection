import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "versions/v2-Small.pt"
CONFIDENCE_THRESHOLD = 0.45
FIRE_CLASS_ID = 1

VALID_DIR = Path(r"D:\Учеба\Практики\семестр_2\летняя практика\ROBOFLOW_FINAL_DATASET\valid")
OUTPUT_DIR = Path("predictions_valid")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

model = YOLO(MODEL_PATH)


def find_images_dir(valid_dir: Path) -> Path:
    candidate = valid_dir / "images"
    if candidate.exists():
        return candidate
    return valid_dir


def draw_fire_boxes(frame, results):
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id != FIRE_CLASS_ID:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"fire {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def main():
    images_dir = find_images_dir(VALID_DIR)
    if not images_dir.exists():
        print(f"Папка с изображениями не найдена: {images_dir}")
        return

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"В папке {images_dir} не найдено изображений.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Найдено изображений: {len(image_paths)}")
    print(f"Результаты будут сохранены в: {OUTPUT_DIR.resolve()}")

    detected_count = 0

    for i, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[{i}/{len(image_paths)}] Не удалось прочитать: {img_path.name}")
            continue

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        has_fire = any(int(b.cls[0]) == FIRE_CLASS_ID for b in results[0].boxes)
        if has_fire:
            detected_count += 1

        frame = draw_fire_boxes(frame, results)

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), frame)

        print(f"[{i}/{len(image_paths)}] {img_path.name} — "
              f"{'обнаружено' if has_fire else 'нет объектов'}")

    print("\nГотово.")
    print(f"Всего изображений: {len(image_paths)}")
    print(f"С обнаруженным огнём: {detected_count}")
    print(f"Без обнаружений: {len(image_paths) - detected_count}")


if __name__ == "__main__":
    main()
