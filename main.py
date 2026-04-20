import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
from rembg import remove, new_session
from pathlib import Path
import numpy as np
import os


class ObjectPickerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Object Picker - Background Removal")
        self.root.geometry("1000x800")

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

        self.resize_timer: str | None = None

        # Model lazy-loading state
        self.rembg_session = None
        self.sam_predictor = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        # Top panel
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=12).pack(side=tk.LEFT, padx=10)

        # Dropdown for model selection
        tk.Label(btn_frame, text="Engine:").pack(side=tk.LEFT, padx=(10, 2))
        self.engine_var = tk.StringVar(value="rembg (isnet)")
        self.engine_dropdown = ttk.Combobox(
            btn_frame,
            textvariable=self.engine_var,
            values=["rembg (isnet)", "SAM (vit_b)"],
            state="readonly",
            width=15
        )
        self.engine_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Extract Selection", command=self.process_selection, width=15).pack(side=tk.LEFT,
                                                                                                      padx=10)

        self.top_status = tk.Label(btn_frame, text="Load an image to begin.", fg="gray")
        self.top_status.pack(side=tk.RIGHT, padx=10)

        # Status Bar at the bottom
        self.status_bar = tk.Label(self.root, text="Ready | Selection: 0x0 | Image: 0x0", bd=1, relief=tk.SUNKEN,
                                   anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Main canvas
        self.canvas = tk.Canvas(self.root, bg="#2b2b2b", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", self.on_window_resize)

    def load_rembg(self) -> bool:
        if self.rembg_session is None:
            self.top_status.config(text="Loading rembg model... please wait.")
            self.root.update()
            try:
                self.rembg_session = new_session("isnet-general-use")
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load rembg:\n{e}")
                return False
        return True

    def load_sam(self) -> bool:
        if self.sam_predictor is None:
            self.top_status.config(text="Loading SAM model... please wait.")
            self.root.update()

            checkpoint_path = "sam_vit_b_01ec64.pth"
            if not os.path.exists(checkpoint_path):
                messagebox.showerror("Missing Weights",
                                     f"SAM checkpoint not found!\nPlease download '{checkpoint_path}' and place it in the same directory as this script.")
                self.top_status.config(text="Error: Missing SAM weights.")
                return False

            try:
                import torch
                from segment_anything import sam_model_registry, SamPredictor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                sam.to(device=device)
                self.sam_predictor = SamPredictor(sam)
            except ImportError:
                messagebox.showerror("Import Error",
                                     "Missing libraries! Run:\npip install torch torchvision segment-anything opencv-python")
                return False
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load SAM:\n{e}")
                return False

        return True

    def update_status_bar(self, selection_w: int = 0, selection_h: int = 0) -> None:
        if self.original_img:
            img_w, img_h = self.original_img.size
            self.status_bar.config(
                text=f"Ready | Selection: {selection_w}x{selection_h} px | Image Size: {img_w}x{img_h} px")
        else:
            self.status_bar.config(text="Ready | Selection: 0x0 | Image: 0x0")

    def load_image(self) -> None:
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filepath: return

        self.img_path = Path(filepath)
        self.original_img = Image.open(self.img_path).convert("RGBA")

        self.bbox = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        self.display_image()
        self.top_status.config(text=f"Loaded: {self.img_path.name}")
        self.update_status_bar()

    def display_image(self) -> None:
        if not self.original_img: return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 600

        img_w, img_h = self.original_img.size
        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        if self.scale_factor > 1.0: self.scale_factor = 1.0

        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        resized_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        self.img_x = (canvas_w - new_w) // 2
        self.img_y = (canvas_h - new_h) // 2

        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.display_img)

        if self.bbox: self.draw_scaled_bbox()

    def on_window_resize(self, event: tk.Event) -> None:
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(100, self.display_image)

    def draw_scaled_bbox(self) -> None:
        if not self.bbox: return
        x1, y1, x2, y2 = self.bbox
        cx1 = int(x1 * self.scale_factor) + self.img_x
        cy1 = int(y1 * self.scale_factor) + self.img_y
        cx2 = int(x2 * self.scale_factor) + self.img_x
        cy2 = int(y2 * self.scale_factor) + self.img_y

        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="cyan", width=2, dash=(4, 4))

    def get_real_coords(self, cx: int, cy: int) -> tuple[int, int]:
        rx = int((cx - self.img_x) / self.scale_factor)
        ry = int((cy - self.img_y) / self.scale_factor)
        return rx, ry

    def on_press(self, event: tk.Event) -> None:
        if not self.display_img: return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="cyan", width=2, dash=(4, 4))

    def on_drag(self, event: tk.Event) -> None:
        if not self.display_img or not self.rect_id: return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
        rx1, ry1 = self.get_real_coords(self.start_x, self.start_y)
        rx2, ry2 = self.get_real_coords(event.x, event.y)
        self.update_status_bar(abs(rx2 - rx1), abs(ry2 - ry1))

    def on_release(self, event: tk.Event) -> None:
        if not self.display_img or not self.rect_id: return
        end_x, end_y = event.x, event.y
        x1, y1 = self.get_real_coords(self.start_x, self.start_y)
        x2, y2 = self.get_real_coords(end_x, end_y)

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        img_w, img_h = self.original_img.size
        # No extra padding for SAM, it prefers tight boxes. Padding kept for rembg.
        padding = 40 if self.engine_var.get() == "rembg (isnet)" else 0
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        self.bbox = (x1, y1, x2, y2)
        self.update_status_bar(x2 - x1, y2 - y1)

    def process_selection(self) -> None:
        if not self.original_img or not self.img_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if not self.bbox or (self.bbox[2] - self.bbox[0] < 10):
            messagebox.showwarning("Warning", "Please draw a valid bounding box.")
            return

        engine = self.engine_var.get()
        self.top_status.config(text=f"Processing with {engine}...")
        self.root.update()

        try:
            if engine == "rembg (isnet)":
                if not self.load_rembg(): return
                cropped_img = self.original_img.crop(self.bbox)
                output_img = remove(cropped_img, session=self.rembg_session)

            elif engine == "SAM (vit_b)":
                if not self.load_sam(): return

                # 1. Convert PIL to RGB numpy array for SAM
                image_array = np.array(self.original_img.convert("RGB"))
                self.sam_predictor.set_image(image_array)

                # 2. Format bounding box for SAM
                input_box = np.array(self.bbox)

                # 3. Predict mask based on the box
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                # 4. Convert the boolean mask back into a PIL Image alpha channel
                mask_array = (masks[0] * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_array).convert("L")

                # Apply mask to the original image
                output_img = self.original_img.copy()
                output_img.putalpha(mask_img)

                # Crop to the bounding box so the output isn't the size of the whole original image
                output_img = output_img.crop(self.bbox)

            self.show_approval_window(output_img)
            self.top_status.config(text="Waiting for user approval...")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
            self.top_status.config(text="Error during processing.")

    def show_approval_window(self, extracted_img: Image.Image) -> None:
        top = tk.Toplevel(self.root)
        top.title("Review & Background Selection")
        top.geometry("800x800")
        top.grab_set()

        self.selected_bg_color: tuple[int, int, int] | None = None

        ctrl_frame = tk.Frame(top, pady=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(ctrl_frame, text="Select Background:").pack(side=tk.LEFT, padx=10)

        preview_canvas = tk.Canvas(top, bg="#2b2b2b")
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        def render_preview() -> None:
            if self.selected_bg_color is None:
                bg = Image.new("RGBA", extracted_img.size, (200, 200, 200, 255))
                for x in range(0, bg.width, 20):
                    for y in range(0, bg.height, 20):
                        if (x // 20 + y // 20) % 2 == 0:
                            bg.paste((255, 255, 255, 255), [x, y, x + 20, y + 20])
                final_preview = Image.alpha_composite(bg, extracted_img)
            else:
                bg = Image.new("RGBA", extracted_img.size, self.selected_bg_color + (255,))
                final_preview = Image.alpha_composite(bg, extracted_img)

            canvas_w = preview_canvas.winfo_width()
            canvas_h = preview_canvas.winfo_height()
            if canvas_w < 10: canvas_w, canvas_h = 750, 600

            scale = min(canvas_w / final_preview.width, canvas_h / final_preview.height)
            if scale < 1.0:
                new_w, new_h = int(final_preview.width * scale), int(final_preview.height * scale)
                final_preview = final_preview.resize((new_w, new_h), Image.Resampling.LANCZOS)

            tk_preview = ImageTk.PhotoImage(final_preview)
            preview_canvas.delete("all")
            preview_canvas.image = tk_preview
            preview_canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=tk_preview)

        # Re-render on window resize to keep it centered and scaled
        preview_canvas.bind("<Configure>", lambda e: render_preview())

        def set_bg_transparent() -> None:
            self.selected_bg_color = None; render_preview()

        def set_bg_white() -> None:
            self.selected_bg_color = (255, 255, 255); render_preview()

        def set_bg_black() -> None:
            self.selected_bg_color = (0, 0, 0); render_preview()

        def set_bg_custom() -> None:
            color = colorchooser.askcolor(title="Choose background color")
            if color[0]:
                self.selected_bg_color = tuple(int(c) for c in color[0])
                render_preview()

        tk.Button(ctrl_frame, text="Transparent", command=set_bg_transparent).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="White", command=set_bg_white).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Black", command=set_bg_black).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Custom...", command=set_bg_custom).pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(top)
        btn_frame.pack(side=tk.BOTTOM, pady=20)

        def approve() -> None:
            out_dir = self.img_path.parent / "masked_photos"
            out_dir.mkdir(exist_ok=True)

            if self.selected_bg_color is None:
                save_path = out_dir / f"{self.img_path.stem}_extracted.png"
                extracted_img.save(save_path)
            else:
                save_path = out_dir / f"{self.img_path.stem}_extracted.jpg"
                bg = Image.new("RGBA", extracted_img.size, self.selected_bg_color + (255,))
                final_img = Image.alpha_composite(bg, extracted_img).convert("RGB")
                final_img.save(save_path, quality=95)

            self.top_status.config(text=f"Saved: {save_path.name}")
            top.destroy()

        def discard() -> None:
            self.top_status.config(text="Discarded.")
            top.destroy()

        tk.Button(btn_frame, text="✅ Save", command=approve, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                  width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ Discard", command=discard, bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                  width=15).pack(side=tk.RIGHT, padx=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectPickerApp(root)
    root.mainloop()