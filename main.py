import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

MODEL_PATH = "versions/v2-Small.pt"
CONFIDENCE_THRESHOLD = 0.45  
FIRE_CLASS_ID = 1       

model = YOLO(MODEL_PATH)


def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


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
    video_path = select_video_file()
    if not video_path:
        print("Файл не выбран. Выход.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео.")
        return

    print("Нажмите 'q' в окне видео, чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось.")
            break

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        frame = draw_fire_boxes(frame, results)

        cv2.imshow("Fire Detection - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()