import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove, new_session
from pathlib import Path


class ObjectPickerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Object Picker - Background Removal")
        self.root.geometry("900x700")

        print("Loading AI model...")
        # Switched to isnet-general-use which often handles object edges better than u2net
        # Note: rembg will download this model on first run!
        self.session = new_session("isnet-general-use")
        print("Model loaded.")

        # State variables
        self.img_path: Path | None = None
        self.original_img: Image.Image | None = None
        self.display_img: ImageTk.PhotoImage | None = None
        self.scale_factor: float = 1.0

        # Bounding box state
        self.start_x: int = 0
        self.start_y: int = 0
        self.rect_id: int | None = None
        self.bbox: tuple[int, int, int, int] | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Extract Selection", command=self.process_selection, width=15).pack(side=tk.LEFT,
                                                                                                      padx=10)

        self.status_label = tk.Label(btn_frame, text="Load an image to begin.", fg="gray")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(self.root, bg="#2b2b2b", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self) -> None:
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filepath:
            return

        self.img_path = Path(filepath)
        self.original_img = Image.open(self.img_path).convert("RGBA")

        self.bbox = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        self.display_image()
        self.status_label.config(text=f"Loaded: {self.img_path.name} | Draw a box around the object.")

    def display_image(self) -> None:
        if not self.original_img: return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 600

        img_w, img_h = self.original_img.size
        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)

        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        resized_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        self.img_x = (canvas_w - new_w) // 2
        self.img_y = (canvas_h - new_h) // 2

        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.display_img)

    def on_press(self, event: tk.Event) -> None:
        if not self.display_img: return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="cyan", width=2, dash=(4, 4))

    def on_drag(self, event: tk.Event) -> None:
        if not self.display_img or not self.rect_id: return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        if not self.display_img or not self.rect_id: return
        end_x, end_y = event.x, event.y

        x1 = int((min(self.start_x, end_x) - self.img_x) / self.scale_factor)
        y1 = int((min(self.start_y, end_y) - self.img_y) / self.scale_factor)
        x2 = int((max(self.start_x, end_x) - self.img_x) / self.scale_factor)
        y2 = int((max(self.start_y, end_y) - self.img_y) / self.scale_factor)

        # Apply your idea: padding for better context
        padding = 40
        img_w, img_h = self.original_img.size

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        self.bbox = (x1, y1, x2, y2)

    def process_selection(self) -> None:
        if not self.original_img or not self.img_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if not self.bbox or (self.bbox[2] - self.bbox[0] < 10) or (self.bbox[3] - self.bbox[1] < 10):
            messagebox.showwarning("Warning", "Please draw a valid bounding box around the object.")
            return

        self.status_label.config(text="Processing... please wait.")
        self.root.update()

        try:
            cropped_img = self.original_img.crop(self.bbox)
            output_img = remove(cropped_img, session=self.session)

            # Send to approval window instead of saving immediately
            self.show_approval_window(output_img)
            self.status_label.config(text="Waiting for user approval...")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
            self.status_label.config(text="Error during processing.")

    def show_approval_window(self, img: Image.Image) -> None:
        top = tk.Toplevel(self.root)
        top.title("Review Extraction")
        top.geometry("650x750")
        top.grab_set()  # Forces focus on this window

        checkered = Image.new("RGBA", img.size, (200, 200, 200, 255))
        for x in range(0, img.width, 20):
            for y in range(0, img.height, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    checkered.paste((255, 255, 255, 255), [x, y, x + 20, y + 20])

        checkered.alpha_composite(img)

        if img.width > 600 or img.height > 600:
            checkered.thumbnail((600, 600), Image.Resampling.LANCZOS)

        tk_preview = ImageTk.PhotoImage(checkered)
        lbl = tk.Label(top, image=tk_preview)
        lbl.image = tk_preview
        lbl.pack(padx=10, pady=20, expand=True)

        btn_frame = tk.Frame(top)
        btn_frame.pack(side=tk.BOTTOM, pady=20)

        def approve() -> None:
            out_dir = self.img_path.parent / "masked_photos"
            out_dir.mkdir(exist_ok=True)
            save_path = out_dir / f"{self.img_path.stem}_extracted.png"
            img.save(save_path)
            self.status_label.config(text=f"Saved: {save_path.name}")
            top.destroy()

        def discard() -> None:
            self.status_label.config(text="Discarded. Try drawing a different box.")
            top.destroy()

        tk.Button(btn_frame, text="✅ Approve & Save", command=approve, bg="#4CAF50", fg="white",
                  font=("Arial", 12, "bold"), width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ Discard", command=discard, bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                  width=15).pack(side=tk.RIGHT, padx=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectPickerApp(root)
    root.mainloop()