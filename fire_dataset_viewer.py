import os
import sys
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Нужна библиотека Pillow.")
    sys.exit(1)

STATS_CATEGORIES = [
    "Лес",
    "Город",
    "Помещение",
    "Ночь",
    "Дальний план",
    "Ближний план",
    "Техника",
    "Сложный случай",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMG_W, MAX_IMG_H = 900, 620


class FireDatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Dataset Selector")
        self.root.geometry("1400x900")

        self.root_dir = filedialog.askdirectory(
            title="Выберите корневую папку датасета (там, где final_dataset)")
        if not self.root_dir:
            sys.exit()

        self.images_src = filedialog.askdirectory(
            title="Выберите папку с изображениями (например JPEGImages)")
        if not self.images_src:
            sys.exit()

        self.labels_src = filedialog.askdirectory(
            title="Выберите папку с метками YOLO (например txt)")
        if not self.labels_src:
            sys.exit()

        self.split = "val" if messagebox.askyesno(
            "Режим отбора",
            "Это отбор для VALIDATION (val)?\n\n"
            "Да — val (скрипт автоматически исключит фото, "
            "уже добавленные в train по имени файла)\n"
            "Нет — train"
        ) 
        else "train"

        self.target_count = 200 if self.split == "val" else 800

        self.final_images = os.path.join(self.root_dir, "final_dataset", self.split, "images")
        self.final_labels = os.path.join(self.root_dir, "final_dataset", self.split, "labels")
        os.makedirs(self.final_images, exist_ok=True)
        os.makedirs(self.final_labels, exist_ok=True)

        self.root.title(f"Fire Dataset Selector — {self.split.upper()}")

        self.stats_file = os.path.join(self.root_dir, "final_dataset", f"stats_{self.split}.txt")
        self.log_file = os.path.join(self.root_dir, "final_dataset", f"reviewed_log_{self.split}.txt")

        all_images = sorted(
            f for f in os.listdir(self.images_src) if f.lower().endswith(IMG_EXTS)
        )
        self.reviewed = self.load_reviewed()

        self.train_filenames = set()
        if self.split == "val":
            train_images_dir = os.path.join(self.root_dir, "final_dataset", "train", "images")
            if os.path.isdir(train_images_dir):
                self.train_filenames = set(os.listdir(train_images_dir))

        self.image_list = [
            f for f in all_images
            if f not in self.reviewed and f not in self.train_filenames
        ]
        self.total_source = len(all_images)
        self.excluded_as_train_dupe = len(
            [f for f in all_images if f in self.train_filenames]
        )

        self.stats, self.approved_count, self.skipped_count = self.load_stats()

        self.index = 0
        self.stat_labels = {}
        self.last_action = None

        self.build_ui()
        self.show_current_image()

    def load_stats(self):
        stats = {cat: 0 for cat in STATS_CATEGORIES}
        approved = 0
        skipped = 0
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    try:
                        v = int(v)
                    except ValueError:
                        continue
                    if k in stats:
                        stats[k] = v
                    elif k == "Одобрено":
                        approved = v
                    elif k == "Пропущено":
                        skipped = v
        return stats, approved, skipped

    def save_stats(self):
        with open(self.stats_file, "w", encoding="utf-8") as f:
            for cat in STATS_CATEGORIES:
                f.write(f"{cat}={self.stats[cat]}\n")
            f.write(f"Одобрено={self.approved_count}\n")
            f.write(f"Пропущено={self.skipped_count}\n")

    def load_reviewed(self):
        reviewed = set()
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    reviewed.add(line.strip())
        return reviewed

    def mark_reviewed(self, filename):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(filename + "\n")


    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=280, bg="#f0f0f0")
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="Статистика датасета", bg="#f0f0f0",
                  font=("Segoe UI", 13, "bold")).pack(pady=(15, 10))

        self.progress_label = tk.Label(left, text="", bg="#f0f0f0",
                                        font=("Segoe UI", 10))
        self.progress_label.pack(pady=(0, 15))

        for cat in STATS_CATEGORIES:
            row = tk.Frame(left, bg="#f0f0f0")
            row.pack(fill="x", padx=15, pady=4)

            name_lbl = tk.Label(row, text=cat, bg="#f0f0f0",
                                 font=("Segoe UI", 10), anchor="w", width=14)
            name_lbl.pack(side="left")

            count_lbl = tk.Label(row, text=str(self.stats[cat]), bg="#f0f0f0",
                                  font=("Segoe UI", 10, "bold"), width=3)
            count_lbl.pack(side="left")
            self.stat_labels[cat] = count_lbl

            plus_btn = tk.Button(row, text="+", width=2, font=("Segoe UI", 9, "bold"),
                                  command=lambda c=cat: self.increment_stat(c))
            plus_btn.pack(side="left", padx=(6, 0))

        sep = tk.Frame(left, height=1, bg="#c0c0c0")
        sep.pack(fill="x", padx=10, pady=15)

        self.approved_label = tk.Label(left, text="", bg="#f0f0f0",
                                        font=("Segoe UI", 10), fg="#2e7d32")
        self.approved_label.pack(pady=2)
        self.skipped_label = tk.Label(left, text="", bg="#f0f0f0",
                                       font=("Segoe UI", 10), fg="#c62828")
        self.skipped_label.pack(pady=2)

        undo_btn = tk.Button(left, text="Отменить последнее действие",
                              command=self.undo_last, font=("Segoe UI", 9))
        undo_btn.pack(pady=(20, 0), padx=15, fill="x")

        center = tk.Frame(main, bg="white")
        center.pack(side="left", fill="both", expand=True)

        self.filename_label = tk.Label(center, text="", bg="white",
                                        font=("Segoe UI", 10))
        self.filename_label.pack(pady=(10, 5))

        self.image_label = tk.Label(center, bg="white")
        self.image_label.pack(expand=True)

        btn_frame = tk.Frame(center, bg="white")
        btn_frame.pack(pady=25)

        approve_btn = tk.Button(
            btn_frame, text="✓", command=self.approve,
            bg="#2e7d32", fg="white", font=("Segoe UI", 22, "bold"),
            width=4, height=1, relief="flat", activebackground="#1b5e20",
            activeforeground="white")
        approve_btn.pack(side="left", padx=20)

        skip_btn = tk.Button(
            btn_frame, text="✗", command=self.skip,
            bg="#c62828", fg="white", font=("Segoe UI", 22, "bold"),
            width=4, height=1, relief="flat", activebackground="#8e0000",
            activeforeground="white")
        skip_btn.pack(side="left", padx=20)

        self.root.bind("<Return>", lambda e: self.approve())
        self.root.bind("<BackSpace>", lambda e: self.skip())

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_stat_display()

    def show_current_image(self):
        if self.index >= len(self.image_list):
            self.image_label.config(image="", text="Все изображения просмотрены 🎉",
                                     font=("Segoe UI", 16))
            self.filename_label.config(text="")
            self.update_stat_display()
            return

        filename = self.image_list[self.index]
        path = os.path.join(self.images_src, filename)
        try:
            img = Image.open(path)
            img.thumbnail((MAX_IMG_W, MAX_IMG_H))
            self.current_image_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image_tk, text="")
        except Exception as e:
            self.image_label.config(image="", text=f"Не удалось открыть:\n{e}")

        self.filename_label.config(text=filename)
        self.update_stat_display()

    def update_stat_display(self):
        remaining = max(len(self.image_list) - self.index, 0)
        dupe_note = (f"\nИсключено (уже в train): {self.excluded_as_train_dupe}"
                     if self.split == "val" else "")
        self.progress_label.config(
            text=f"Осталось в очереди: {remaining}\n"
                 f"Всего в источнике: {self.total_source}{dupe_note}")
        self.approved_label.config(
            text=f"Одобрено: {self.approved_count} / {self.target_count}")
        self.skipped_label.config(text=f"Пропущено: {self.skipped_count}")
        for cat in STATS_CATEGORIES:
            self.stat_labels[cat].config(text=str(self.stats[cat]))

    def increment_stat(self, category):
        self.stats[category] += 1
        self.save_stats()
        self.update_stat_display()

    def approve(self):
        if self.index >= len(self.image_list):
            return
        filename = self.image_list[self.index]
        src_img = os.path.join(self.images_src, filename)
        basename, _ = os.path.splitext(filename)
        label_filename = basename + ".txt"
        src_label = os.path.join(self.labels_src, label_filename)

        dst_img = os.path.join(self.final_images, filename)
        dst_label = os.path.join(self.final_labels, label_filename)

        try:
            shutil.copy2(src_img, dst_img)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать фото:\n{e}")
            return

        label_copied = False
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            label_copied = True
        else:
            messagebox.showwarning("Внимание", f"Метка не найдена для {filename}")

        self.approved_count += 1
        self.last_action = ("approve", filename, dst_img, dst_label if label_copied else None)
        self.mark_reviewed(filename)
        self.save_stats()
        self.index += 1
        self.show_current_image()

    def skip(self):
        if self.index >= len(self.image_list):
            return
        filename = self.image_list[self.index]
        self.skipped_count += 1
        self.last_action = ("skip", filename, None, None)
        self.mark_reviewed(filename)
        self.save_stats()
        self.index += 1
        self.show_current_image()

    def undo_last(self):
        if self.last_action is None:
            return
        action_type, filename, dst_img, dst_label = self.last_action

        if action_type == "approve":
            if dst_img and os.path.exists(dst_img):
                os.remove(dst_img)
            if dst_label and os.path.exists(dst_label):
                os.remove(dst_label)
            self.approved_count = max(self.approved_count - 1, 0)
        elif action_type == "skip":
            self.skipped_count = max(self.skipped_count - 1, 0)

        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() != filename]
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        self.index = max(self.index - 1, 0)
        self.last_action = None
        self.save_stats()
        self.show_current_image()

    def on_close(self):
        self.save_stats()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FireDatasetViewer(root)
    root.mainloop()