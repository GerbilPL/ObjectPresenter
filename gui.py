import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import tkinter.scrolledtext as st
from PIL import Image, ImageTk
from pathlib import Path
import traceback
import threading
import datetime

from config_manager import ConfigManager
from image_processor import ImageProcessor
from progress_tracker import ProgressTracker


class ObjectPickerApp:
    """Main GUI application for background removal and inpainting of images.
    
    Supports single image and batch processing workflows with interactive
    bounding box selection, device preference management, and background composition.
    """
    def __init__(self, root: tk.Tk) -> None:
        """Initializes the ObjectPickerApp with GUI components and state.
        
        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Object Picker - Background Removal & Inpainting")
        self.root.geometry("1100x800")
        self.root.minsize(640, 480)

        self.cfg = ConfigManager()
        self.processor = ImageProcessor()
        self.tracker = None

        self.process_error = None
        self.processing_thread = None

        # --- Batch System State ---
        self.is_batch_mode = False
        self.batch_files = []  # Queue of remaining file paths
        self.batch_tasks = []  # Stored tasks: dict with original_img, bbox, path
        self.batch_results = []  # Completed outputs ready for review

        # --- Hardware status ---
        self.has_cuda = None
        self.cuda_warned = False

        # --- Canvas State ---
        self.img_path: Path | None = None
        self.original_img: Image.Image | None = None
        self.display_img: ImageTk.PhotoImage | None = None
        self.scale_factor: float = 1.0

        # Bounding box state
        self.start_x: int = 0
        self.start_y: int = 0
        self.rect_id: int | None = None
        self.margin_rect_id: int | None = None
        self.bbox: tuple[int, int, int, int] | None = None

        self.resize_timer: str | None = None

        self._setup_menu()
        self._setup_ui()
        self.apply_theme()

        # Start HW check in background
        self._check_hardware_async()

    def _check_hardware_async(self):
        """Asynchronously checks for PyTorch CUDA support without blocking UI."""

        def worker():
            try:
                import torch
                cuda_ok = torch.cuda.is_available()
            except Exception:
                cuda_ok = False
            self.root.after(0, lambda: self._update_hw_status(cuda_ok))

        threading.Thread(target=worker, daemon=True).start()

    def _update_hw_status(self, cuda_ok: bool = None):
        """Dynamically updates the HW label based on physical availability AND user preference."""
        if cuda_ok is not None:
            self.has_cuda = cuda_ok

        if self.has_cuda is None:
            return  # Still checking

        pref = self.device_var.get()
        if pref == "CPU":
            text = " CPU (Forced) "
        elif pref == "CUDA":
            if self.has_cuda:
                text = " GPU (Forced) "
            else:
                text = " GPU Unavailable! "
        else:  # Auto
            if self.has_cuda:
                text = " GPU (Auto) "
            else:
                text = " CPU (Auto) "

        self.hw_status_label.config(text=text)

    def _setup_menu(self) -> None:
        """Creates and configures the application menu bar with File and Edit menus."""
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Single Image", command=self.load_single_image)
        file_menu.add_command(label="Load Batch...", command=self.load_batch_images)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Settings", command=self.open_settings)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        self.root.config(menu=menubar)

    def _setup_ui(self) -> None:
        """Builds all UI elements including scrollable toolbar, canvas, and status panels."""
        # --- Top Menu with Scrollbar for resize protection ---
        self.top_scroll_frame = tk.Frame(self.root)
        self.top_scroll_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_canvas = tk.Canvas(self.top_scroll_frame, height=50, highlightthickness=0)
        self.btn_scrollbar = tk.Scrollbar(self.top_scroll_frame, orient=tk.HORIZONTAL, command=self.btn_canvas.xview)

        self.btn_frame = tk.Frame(self.btn_canvas)
        self.btn_frame.bind(
            "<Configure>",
            lambda e: self.btn_canvas.configure(scrollregion=self.btn_canvas.bbox("all"))
        )
        self.btn_canvas.create_window((0, 0), window=self.btn_frame, anchor="nw")
        self.btn_canvas.configure(xscrollcommand=self.btn_scrollbar.set)

        self.btn_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.btn_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Build controls inside the scrollable btn_frame
        tk.Button(self.btn_frame, text="Load Image", command=self.load_single_image, width=12).pack(side=tk.LEFT,
                                                                                                    padx=10, pady=10)

        tk.Label(self.btn_frame, text="Engine:").pack(side=tk.LEFT, padx=(5, 2))
        self.engine_var = tk.StringVar(value="rembg (isnet)")
        self.engine_dropdown = ttk.Combobox(
            self.btn_frame,
            textvariable=self.engine_var,
            values=["rembg (isnet)", "SAM (vit_b)"],
            state="readonly",
            width=12
        )
        self.engine_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Label(self.btn_frame, text="Device:").pack(side=tk.LEFT, padx=(5, 2))
        self.device_var = tk.StringVar(value=self.cfg.get("device_preference", "Auto"))
        self.device_dropdown = ttk.Combobox(
            self.btn_frame, textvariable=self.device_var,
            values=["Auto", "CUDA", "CPU"], state="readonly", width=6
        )
        self.device_dropdown.pack(side=tk.LEFT, padx=5)

        def on_device_change(*args):
            self.cfg.set("device_preference", self.device_var.get())
            self.processor.clear_models()
            self._update_hw_status()

        self.device_dropdown.bind("<<ComboboxSelected>>", on_device_change)

        margin_frame = tk.Frame(self.btn_frame)
        margin_frame.pack(side=tk.LEFT, padx=10)

        self.margin_val_label = tk.Label(margin_frame, text="Margin: 0")
        self.margin_val_label.pack(side=tk.LEFT)

        self.margin_var = tk.IntVar(value=self.cfg.get("default_margin", 20))
        self.margin_slider = tk.Scale(
            margin_frame, from_=-50, to=70, orient=tk.HORIZONTAL,
            variable=self.margin_var, showvalue=False, length=100,
            command=self.on_margin_slider_change # TODO: change to update_margin_visuals and test
        )
        self.margin_slider.pack(side=tk.LEFT, padx=5)

        self.margin_rel_var = tk.BooleanVar(value=self.cfg.get("margin_relative", False))
        self.margin_rel_check = tk.Checkbutton(
            margin_frame, text="% (Rel)", variable=self.margin_rel_var,
            command=self.update_margin_visuals
        )
        self.margin_rel_check.pack(side=tk.LEFT)

        inpaint_frame = tk.Frame(self.btn_frame, highlightbackground="#444", highlightthickness=1)
        inpaint_frame.pack(side=tk.LEFT, padx=10)

        self.inpaint_var = tk.BooleanVar(value=self.cfg.get("inpaint_enabled", False))
        self.inpaint_check = tk.Checkbutton(inpaint_frame, text="Inpainting", variable=self.inpaint_var)
        self.inpaint_check.pack(side=tk.LEFT, padx=5)

        self.inpaint_method_var = tk.StringVar(value=self.cfg.get("inpaint_method", "OpenCV"))
        self.inpaint_dropdown = ttk.Combobox(
            inpaint_frame, textvariable=self.inpaint_method_var,
            values=["OpenCV", "LaMa"], state="readonly", width=8
        )
        self.inpaint_dropdown.pack(side=tk.LEFT, padx=5)

        self.action_btn = tk.Button(self.btn_frame, text="Extract Selection", command=self.on_action_click,
                                    bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.action_btn.pack(side=tk.LEFT, padx=15)  # Changed to LEFT so it flows with scrollbar

        # Global keybind for Enter -> confirms selection
        self.root.bind("<Return>", lambda e: self.on_action_click())

        # --- Progress & Status Frames (Bottom) ---
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_frame = tk.Frame(self.bottom_frame, pady=5)
        self.progress_frame.pack(side=tk.TOP, fill=tk.X)

        self.overall_progress_var = tk.DoubleVar()
        self.item_progress_var = tk.DoubleVar()

        tk.Label(self.progress_frame, text="Item:").pack(side=tk.LEFT, padx=(10, 5))
        self.item_pb = ttk.Progressbar(self.progress_frame, variable=self.item_progress_var, maximum=100, length=200)
        self.item_pb.pack(side=tk.LEFT, padx=5)

        tk.Label(self.progress_frame, text="Batch:").pack(side=tk.LEFT, padx=(15, 5))
        self.overall_pb = ttk.Progressbar(self.progress_frame, variable=self.overall_progress_var, maximum=100,
                                          length=200)
        self.overall_pb.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = tk.Button(self.progress_frame, text="Cancel", command=self.cancel_processing,
                                    state=tk.DISABLED, bg="#f44336", fg="white", width=10)
        self.cancel_btn.pack(side=tk.RIGHT, padx=10)

        # Status Bar split into main status (Left) and HW info (Right)
        self.status_bar_frame = tk.Frame(self.bottom_frame)
        self.status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_bar = tk.Label(self.status_bar_frame, text="Ready | Selection: 0x0 | Image: 0x0", bd=1,
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hw_status_label = tk.Label(self.status_bar_frame, text=" HW: Checking... ", bd=1, relief=tk.SUNKEN,
                                        anchor=tk.E)
        self.hw_status_label.pack(side=tk.RIGHT)

        # --- Canvas ---
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", self.on_window_resize)

        self.on_margin_slider_change(self.margin_var.get())

    def apply_theme(self) -> None:
        """Applies the selected theme (System, Light, Dark, or Time-Based) to all UI elements.
        
        Dynamically adjusts colors based on theme preference and recursively
        updates all child widgets.
        """
        theme = self.cfg.get("theme", "System")

        if theme == "Time-Based":
            now = datetime.datetime.now().time()
            try:
                light_t = datetime.datetime.strptime(self.cfg.get("time_light_hour", "09:00"), "%H:%M").time()
                dark_t = datetime.datetime.strptime(self.cfg.get("time_dark_hour", "18:00"), "%H:%M").time()
            except ValueError:
                light_t = datetime.time(9, 0)
                dark_t = datetime.time(18, 0)

            if light_t < dark_t:
                is_light = light_t <= now < dark_t
            else:
                is_light = now >= light_t or now < dark_t
            theme = "Light" if is_light else "Dark"

        if theme == "System":
            import sys
            if sys.platform == "win32":
                try:
                    import winreg
                    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                    key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    theme = "Light" if val == 1 else "Dark"
                except:
                    theme = "Dark"
            else:
                theme = "Dark"

        if theme == "Dark":
            bg_main, bg_sec, fg_main = "#1e1e1e", "#2b2b2b", "#ffffff"
        else:
            bg_main, bg_sec, fg_main = "#f0f0f0", "#e0e0e0", "#000000"

        self.root.configure(bg=bg_main)
        self.top_scroll_frame.configure(bg=bg_main)
        self.btn_canvas.configure(bg=bg_main)
        self.btn_frame.configure(bg=bg_main)
        self.canvas.configure(bg=bg_sec)

        def recursive_theme(widget):
            try:
                if isinstance(widget, tk.Frame):
                    widget.configure(bg=bg_main)
                elif isinstance(widget, tk.Label):
                    widget.configure(bg=bg_main, fg=fg_main)
                elif isinstance(widget, tk.Checkbutton):
                    widget.configure(bg=bg_main, fg=fg_main, selectcolor=bg_sec)
                elif isinstance(widget, tk.Scale):
                    widget.configure(bg=bg_main, highlightthickness=0)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                if not isinstance(child, ttk.Progressbar):
                    recursive_theme(child)

        recursive_theme(self.btn_frame)
        recursive_theme(self.bottom_frame)

    def open_settings(self) -> None:
        """Opens a modal settings dialog for theme, timing, and filename template configuration."""
        top = tk.Toplevel(self.root)
        top.title("Settings")
        # Removed hardcoded geometry to let it auto-wrap elements natively
        top.resizable(False, False)
        top.grab_set()

        def on_theme_change(*args):
            self.cfg.set("theme", theme_var.get())
            self.apply_theme()
            if theme_var.get() == "Time-Based":
                time_frame.grid()
            else:
                time_frame.grid_remove()

        tk.Label(top, text="Theme:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        theme_var = tk.StringVar(value=self.cfg.get("theme"))
        theme_cb = ttk.Combobox(top, textvariable=theme_var, values=["System", "Light", "Dark", "Time-Based"],
                                state="readonly")
        theme_cb.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        theme_cb.bind("<<ComboboxSelected>>", on_theme_change)

        # Time-Based Options
        time_frame = tk.Frame(top)
        tk.Label(time_frame, text="Light Mode starts (HH:MM):").grid(row=0, column=0, sticky="e", pady=2)
        light_time_var = tk.StringVar(value=self.cfg.get("time_light_hour", "09:00"))
        tk.Entry(time_frame, textvariable=light_time_var, width=10).grid(row=0, column=1, sticky="w", padx=5)

        tk.Label(time_frame, text="Dark Mode starts (HH:MM):").grid(row=1, column=0, sticky="e", pady=2)
        dark_time_var = tk.StringVar(value=self.cfg.get("time_dark_hour", "18:00"))
        tk.Entry(time_frame, textvariable=dark_time_var, width=10).grid(row=1, column=1, sticky="w", padx=5)

        time_frame.grid(row=1, column=0, columnspan=2, pady=5)
        if theme_var.get() != "Time-Based":
            time_frame.grid_remove()

        # Filename Template
        template_frame = tk.Frame(top)
        template_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Label(template_frame, text="Output Filename Template:").pack(side=tk.TOP, anchor="w", padx=10)

        input_frame = tk.Frame(template_frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        template_var = tk.StringVar(value=self.cfg.get("filename_template"))
        template_entry = tk.Entry(input_frame, textvariable=template_var, width=30)
        template_entry.pack(side=tk.LEFT)

        def show_template_help():
            help_text = (
                "Available template variables:\n\n"
                "filename$ - Original filename without extension\n"
                "date$ - Current date (YYYY-MM-DD)\n"
                "time$ - Current time (HHMMSS)\n"
                "engine$ - Selected AI Engine\n"
                "bg$ - Background (transparent, hex, or image)"
            )
            messagebox.showinfo("Filename Templates Help", help_text, parent=top)

        tk.Button(input_frame, text="❓", command=show_template_help, relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

        def save_settings() -> None:
            self.cfg.set("theme", theme_var.get())
            self.cfg.set("time_light_hour", light_time_var.get())
            self.cfg.set("time_dark_hour", dark_time_var.get())
            self.cfg.set("filename_template", template_var.get())
            self.apply_theme()
            top.destroy()

        tk.Button(top, text="Save Settings & Close", command=save_settings, width=20).grid(row=3, column=0,
                                                                                           columnspan=2, pady=20)

    def handle_inpaint_error(self, err_msg: str, module_name: str) -> bool:
        """Displays an error dialog when inpainting module initialization fails.
        
        Args:
            err_msg: Error message or traceback to display.
            module_name: Name of the failed inpainting module.
            
        Returns:
            True if user chooses to continue, False if user aborts.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Module Load Error")
        dialog.geometry("650x450")
        dialog.grab_set()

        theme = self.cfg.get("theme", "System")
        bg_color = "#1e1e1e" if theme == "Dark" or (
                    theme == "System" and self.root.cget('bg') == "#1e1e1e") else "#f0f0f0"
        fg_color = "#ffffff" if bg_color == "#1e1e1e" else "#000000"
        dialog.configure(bg=bg_color)

        header_frame = tk.Frame(dialog, bg=bg_color)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header_frame, text=f"⚠️ An error occurred when initializing module: {module_name}.",
                 font=("Arial", 12, "bold"), fg="#f44336", bg=bg_color).pack(anchor="w")
        tk.Label(header_frame, text="You can abort or continue this process without inpainting", bg=bg_color,
                 fg=fg_color).pack(anchor="w")

        text_area = st.ScrolledText(dialog, wrap=tk.WORD, height=12, font=("Consolas", 9),
                                    bg="#2b2b2b" if bg_color == "#1e1e1e" else "#ffffff",
                                    fg="#ff7b72" if bg_color == "#1e1e1e" else "#d73a49")
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        text_area.insert(tk.END, err_msg)
        text_area.configure(state=tk.DISABLED)

        disable_inpaint_var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(dialog, text="Turn off inpainting for this session (for bulk processing).",
                             variable=disable_inpaint_var, bg=bg_color, fg=fg_color,
                             selectcolor="#2b2b2b" if bg_color == "#1e1e1e" else "#ffffff")
        chk.pack(anchor="w", padx=10, pady=5)

        self.continue_flag = False
        btn_frame = tk.Frame(dialog, bg=bg_color)
        btn_frame.pack(fill=tk.X, padx=10, pady=15)

        def on_continue() -> None:
            self.continue_flag = True
            if disable_inpaint_var.get():
                self.inpaint_var.set(False)
                self.cfg.set("inpaint_enabled", False)
            dialog.destroy()

        def on_abort() -> None:
            self.continue_flag = False
            dialog.destroy()

        tk.Button(btn_frame, text="Continue without inpainting", command=on_continue, bg="#2196F3", fg="white",
                  width=25).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Abort", command=on_abort, width=15).pack(side=tk.RIGHT, padx=5)

        self.root.wait_window(dialog)
        return self.continue_flag

    def update_status_text(self, text: str) -> None:
        """Updates the status bar text if the widget exists.
        
        Args:
            text: Status message to display.
        """
        if self.status_bar.winfo_exists():
            self.status_bar.config(text=text)

    # --- CANVAS LOGIC & IMAGE LOADING ---

    def calculate_margin_px(self) -> int:
        """Calculates the margin in pixels based on current margin settings and bbox.
        
        Returns:
            Margin in pixels (absolute or relative based on margin_rel_var).
        """
        if not self.bbox: return 0
        val = self.margin_var.get()
        if self.margin_rel_var.get():
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            return int(max(w, h) * (val / 100.0))
        return val

    def on_margin_slider_change(self, val) -> None:
        """Handles margin slider value changes and updates the display label.
        
        Args:
            val: New margin slider value.
        """
        suffix = "%" if self.margin_rel_var.get() else "px"
        self.margin_val_label.config(text=f"Margin: {self.margin_var.get()}{suffix}")
        self.update_margin_visuals()

    def update_margin_visuals(self, *args) -> None:
        """Updates the margin rectangle visualization on the canvas.
        
        Redraws the dashed rectangle showing the expanded bounding box with margin.
        """
        suffix = "%" if self.margin_rel_var.get() else "px"
        self.margin_val_label.config(text=f"Margin: {self.margin_var.get()}{suffix}")

        if self.margin_rect_id:
            self.canvas.delete(self.margin_rect_id)
            self.margin_rect_id = None

        if not self.bbox or not self.original_img: return

        margin_px = self.calculate_margin_px()
        x1, y1, x2, y2 = self.bbox

        mx1 = max(0, x1 - margin_px)
        my1 = max(0, y1 - margin_px)
        mx2 = min(self.original_img.width, x2 + margin_px)
        my2 = min(self.original_img.height, y2 + margin_px)

        cx1 = int(mx1 * self.scale_factor) + self.img_x
        cy1 = int(my1 * self.scale_factor) + self.img_y
        cx2 = int(mx2 * self.scale_factor) + self.img_x
        cy2 = int(my2 * self.scale_factor) + self.img_y

        self.margin_rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#ff9800", width=2, dash=(2, 4))

    def update_status_bar(self, selection_w: int = 0, selection_h: int = 0) -> None:
        """Updates the status bar with current selection and image dimensions.
        
        Args:
            selection_w: Width of the current selection in pixels (default: 0).
            selection_h: Height of the current selection in pixels (default: 0).
        """
        if self.original_img:
            img_w, img_h = self.original_img.size
            self.status_bar.config(
                text=f"Ready | Selection: {selection_w}x{selection_h} px | Image Size: {img_w}x{img_h} px")
        else:
            self.status_bar.config(text="Ready | Selection: 0x0 | Image: 0x0")

    def _clear_canvas_overlays(self):
        """Clears all bounding box and margin rectangle overlays from the canvas."""
        self.bbox = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.margin_rect_id:
            self.canvas.delete(self.margin_rect_id)
            self.margin_rect_id = None

    def load_single_image(self) -> None:
        """Opens file dialog to load a single image and displays it on canvas."""
        filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filepath: return
        self.is_batch_mode = False
        self.batch_tasks = []
        self.batch_results = []
        self.action_btn.config(text="Extract Selection")

        self.img_path = Path(filepath)
        self.original_img = Image.open(self.img_path).convert("RGBA")
        self._clear_canvas_overlays()
        self.display_image()
        self.update_status_bar()

    def load_batch_images(self) -> None:
        """Opens file dialog to load multiple images for batch processing."""
        filepaths = filedialog.askopenfilenames(title="Select multiple images for batch processing",
                                                filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not filepaths: return

        self.is_batch_mode = True
        self.batch_files = [Path(p) for p in filepaths]
        self.batch_tasks = []
        self.batch_results = []
        self.load_next_batch_image()

    def load_next_batch_image(self) -> None:
        """Loads the next image from the batch queue or starts processing if queue is empty.
        
        Updates the UI with batch progress information.
        """
        if not self.batch_files:
            self.start_processing()
            return

        self.img_path = self.batch_files.pop(0)
        self.original_img = Image.open(self.img_path).convert("RGBA")
        self._clear_canvas_overlays()
        self.display_image()
        self.update_status_bar()

        total_in_batch = len(self.batch_tasks) + len(self.batch_files) + 1
        current_num = len(self.batch_tasks) + 1
        self.action_btn.config(text=f"Next Image ({current_num}/{total_in_batch})")
        self.update_status_text(f"Batch Mode: Draw bounding box for {self.img_path.name} and press Enter")

    def display_image(self) -> None:
        """Renders the current image on canvas, scaled to fit the viewport.
        
        Recalculates scale factor and redraws any active bounding box overlays.
        """
        if not self.original_img: return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w, canvas_h = 800, 600

        img_w, img_h = self.original_img.size
        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        if self.scale_factor > 1.0: self.scale_factor = 1.0

        new_w, new_h = int(img_w * self.scale_factor), int(img_h * self.scale_factor)
        resized_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        self.img_x = (canvas_w - new_w) // 2
        self.img_y = (canvas_h - new_h) // 2
        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.display_img)

        if self.bbox:
            self.draw_scaled_bbox()
            self.update_margin_visuals()

    def on_window_resize(self, event: tk.Event) -> None:
        """Handles canvas resize events with debouncing to avoid excessive redraws.
        
        Args:
            event: The resize event from Tkinter.
        """
        if self.resize_timer: self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(100, self.display_image)

    def draw_scaled_bbox(self) -> None:
        """Draws the scaled bounding box on the canvas at current scale factor."""
        if not self.bbox: return
        x1, y1, x2, y2 = self.bbox
        cx1 = int(x1 * self.scale_factor) + self.img_x
        cy1 = int(y1 * self.scale_factor) + self.img_y
        cx2 = int(x2 * self.scale_factor) + self.img_x
        cy2 = int(y2 * self.scale_factor) + self.img_y

        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00e5ff", width=2, dash=(4, 4))

    def get_real_coords(self, cx: int, cy: int) -> tuple[int, int]:
        """Converts canvas pixel coordinates to original image coordinates.
        
        Accounts for image centering, scaling, and translation on canvas.
        Used to map mouse events to bounding box coordinates in original image space.
        
        Args:
            cx: Canvas pixel X coordinate (0 = left edge of canvas).
            cy: Canvas pixel Y coordinate (0 = top edge of canvas).
        
        Returns:
            Tuple of (image_x, image_y) in original image's coordinate space.
        """
        rx = int((cx - self.img_x) / self.scale_factor)
        ry = int((cy - self.img_y) / self.scale_factor)
        return rx, ry

    def on_press(self, event: tk.Event) -> None:
        """Handles mouse button press to start drawing a bounding box.
        
        Args:
            event: The mouse press event.
        """
        if not self.display_img: return
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="#00e5ff", width=2, dash=(4, 4))

    def on_drag(self, event: tk.Event) -> None:
        """Handles mouse drag to update the bounding box rectangle in real-time.
        
        Args:
            event: The mouse motion event.
        """
        if not self.display_img or not self.rect_id: return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
        rx1, ry1 = self.get_real_coords(self.start_x, self.start_y)
        rx2, ry2 = self.get_real_coords(event.x, event.y)
        self.update_status_bar(abs(rx2 - rx1), abs(ry2 - ry1))

        self.bbox = (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))
        self.update_margin_visuals()

    def on_release(self, event: tk.Event) -> None:
        """Handles mouse button release to finalize the bounding box.
        
        Args:
            event: The mouse release event.
        """
        if not self.display_img or not self.rect_id: return
        rx1, ry1 = self.get_real_coords(self.start_x, self.start_y)
        rx2, ry2 = self.get_real_coords(event.x, event.y)
        self.bbox = (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))
        self.update_status_bar(abs(rx2 - rx1), abs(ry2 - ry1))
        self.update_margin_visuals()

    # --- PROCESSING WIZARD & THREADS ---

    def on_action_click(self) -> None:
        """Handles the main action button click (Extract or Next Image in batch mode).
        
        Validates bounding box and queues the task for processing.
        """
        if not self.original_img or not self.img_path:
            return

        if not self.bbox or (self.bbox[2] - self.bbox[0] < 10):
            messagebox.showwarning("Warning", "Please draw a valid bounding box before confirming.")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            return

        margin_px = self.calculate_margin_px()
        img_w, img_h = self.original_img.size
        working_bbox = (
            max(0, self.bbox[0] - margin_px),
            max(0, self.bbox[1] - margin_px),
            min(img_w, self.bbox[2] + margin_px),
            min(img_h, self.bbox[3] + margin_px)
        )

        task = {
            "path": self.img_path,
            "original_img": self.original_img,
            "working_bbox": working_bbox
        }

        if self.is_batch_mode:
            self.batch_tasks.append(task)
            self.load_next_batch_image()
        else:
            self.batch_tasks = [task]
            self.start_processing()

    def start_processing(self) -> None:
        """Initiates batch processing and launches the worker thread.
        
        Validates hardware availability, preloads inpainting modules, and starts
        the background processing thread.
        """
        if not self.batch_tasks:
            return

        selected_device = self.device_var.get()
        if self.has_cuda is False:
            if selected_device == "CUDA":
                messagebox.showerror("Hardware Error",
                                     "You selected 'CUDA' but PyTorch cannot detect a GPU.\nPlease select 'Auto' or 'CPU', or reinstall PyTorch with CUDA support.")
                return
            if selected_device == "Auto" and not self.cuda_warned:
                ans = messagebox.askyesno("Performance Warning",
                                          "CUDA (GPU) is not available on this system. Processing will use the CPU and may be significantly slower.\n\nDo you want to continue?")
                if not ans:
                    return
                self.cuda_warned = True

        engine = self.engine_var.get()
        inpaint_method = self.inpaint_method_var.get()

        self.cfg.set("default_margin", self.margin_var.get())
        self.cfg.set("margin_relative", self.margin_rel_var.get())
        self.cfg.set("inpaint_enabled", self.inpaint_var.get())
        self.cfg.set("inpaint_method", inpaint_method)

        if self.inpaint_var.get():
            try:
                self.update_status_text(f"Preparing inpainting module: {inpaint_method}...")
                self.processor.preload_inpaint(inpaint_method)
            except Exception as e:
                err_trace = traceback.format_exc()
                continue_without_inpaint = self.handle_inpaint_error(err_trace, inpaint_method)
                if not continue_without_inpaint:
                    self.update_status_text("Aborted by user.")
                    return
                else:
                    self.inpaint_var.set(False)

        self.tracker = ProgressTracker()
        self.tracker.start_batch(len(self.batch_tasks))

        self.cancel_btn.config(state=tk.NORMAL)
        self.process_error = None
        self.batch_results = []

        self.processing_thread = threading.Thread(
            target=self._process_worker,
            args=(engine, self.inpaint_var.get(), inpaint_method, selected_device)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.root.after(100, self._check_progress)

    def _process_worker(self, engine, inpaint_enabled, inpaint_method, device_preference):
        """Worker thread that processes all queued tasks sequentially.
        
        Runs in background thread, iterating over self.batch_tasks, calling
        ImageProcessor.process() for each task, and collecting results into
        self.batch_results. Respects ProgressTracker cancellation flag.
        
        Args:
            engine: The AI engine to use: 'rembg (isnet)' or 'SAM (vit_b)'.
                'rembg' faster, general-purpose; 'SAM' more precise box-guided.
            inpaint_enabled: If True, fills mask holes. If False, outputs extracted
                object with alpha transparency only.
            inpaint_method: When inpaint_enabled=True: 'OpenCV' (fast Telea algorithm)
                or 'LaMa' (neural network, slower but higher quality).
            device_preference: Hardware selection: 'Auto' (auto-detect GPU availability),
                'CUDA' (require GPU, error if unavailable), 'CPU' (force CPU).
        """
        try:
            for idx, task in enumerate(self.batch_tasks):
                if self.tracker.is_cancelled:
                    break
                self.tracker.set_current_item(idx, f"Initializing {task['path'].name}")

                output_img = self.processor.process(
                    original_img=task["original_img"],
                    working_bbox=task["working_bbox"],
                    engine=engine,
                    inpaint_enabled=inpaint_enabled,
                    inpaint_method=inpaint_method,
                    device_preference=device_preference,
                    tracker=self.tracker
                )

                if output_img:
                    self.batch_results.append({
                        "path": task["path"],
                        "result_img": output_img,
                        "engine": engine,
                        "device": device_preference if device_preference != "Auto" else (
                            "CUDA" if self.has_cuda else "CPU"),
                        "bg_type": "color",  # 'color' or 'image'
                        "bg_val": None,  # tuple for color, PIL.Image for image
                        "bg_img_mode": "Center",  # 'Stretch' or 'Center'
                        "reedit": False
                    })

        except InterruptedError:
            self.process_error = "Cancelled"
        except Exception as e:
            traceback.print_exc()
            self.process_error = str(e)

    def _check_progress(self) -> None:
        """Polls progress tracker and updates UI, then schedules next poll.
        
        Runs repeatedly until processing completes, showing item and batch progress.
        """
        if not self.tracker: return

        state = self.tracker.get_state()
        self.overall_progress_var.set(state["overall_progress"])
        self.item_progress_var.set(state["item_progress"])
        self.update_status_text(state["status"])

        if self.processing_thread and self.processing_thread.is_alive():
            self.root.after(100, self._check_progress)
        else:
            self.cancel_btn.config(state=tk.DISABLED)

            if self.process_error == "Cancelled" or state["is_cancelled"]:
                self.update_status_text("Processing cancelled by user.")
                self.overall_progress_var.set(0)
                self.item_progress_var.set(0)
                if len(self.batch_results) > 0:
                    self.show_batch_approval_window()
            elif self.process_error:
                messagebox.showerror("Error", f"Failed to process image:\n{self.process_error}")
                self.update_status_text("Error during processing.")
            elif len(self.batch_results) > 0:
                self.overall_progress_var.set(100)
                self.item_progress_var.set(100)
                self.update_status_text("Processing complete! Waiting for user approval...")
                self.action_btn.config(text="Extract Selection")
                self.show_batch_approval_window()

    def cancel_processing(self) -> None:
        """Signals cancellation to the active processing thread."""
        if self.tracker:
            self.tracker.cancel()
            self.cancel_btn.config(state=tk.DISABLED)

    # --- BATCH REVIEW WINDOW ---

    def _get_output_filename(self, res_dict: dict) -> str:
        """Computes the output filename based on template and result metadata.
        
        Args:
            res_dict: Result dictionary with path, bg_type, bg_val, and engine info.
            
        Returns:
            Generated filename with extension based on background type.
        """
        path = res_dict["path"]
        bg_type = res_dict["bg_type"]
        bg_val = res_dict["bg_val"]
        engine_str = res_dict["engine"].split()[0].lower()

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")

        if bg_type == "color":
            bg_str = "transparent" if bg_val is None else f"{bg_val[0]:02x}{bg_val[1]:02x}{bg_val[2]:02x}"
        else:
            bg_str = "bg-image"

        template = self.cfg.get("filename_template", "filename$_extracted")
        file_stem = template.replace("filename$", path.stem)
        file_stem = file_stem.replace("date$", date_str)
        file_stem = file_stem.replace("time$", time_str)
        file_stem = file_stem.replace("engine$", engine_str)
        file_stem = file_stem.replace("bg$", bg_str)

        ext = ".png" if (bg_type == "color" and bg_val is None) else ".jpg"
        return f"{file_stem}{ext}"

    def show_batch_approval_window(self) -> None:
        """Opens the batch review and background selection modal dialog.
        
        Displays processed images with dual view modes (List/Preview) and allows
        users to select backgrounds, mark for re-edit, and approve/discard results.
        """
        if not self.batch_results:
            return

        top = tk.Toplevel(self.root)
        title = "Review & Background Selection" if not self.is_batch_mode else "Batch Review & Background Selection"
        top.title(title)
        top.geometry("1100x800")
        top.minsize(800, 600)
        top.grab_set()

        self.current_preview_idx = 0

        theme = self.cfg.get("theme", "System")
        is_dark = theme == "Dark" or (theme == "System" and self.root.cget('bg') == "#1e1e1e")
        listbox_fg = "#ffffff" if is_dark else "#000000"
        reedit_fg = "#ff9800" if is_dark else "#d32f2f"

        main_paned = tk.PanedWindow(top, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_paned, width=300)
        right_frame = tk.Frame(main_paned)

        main_paned.add(left_frame, minsize=200)
        main_paned.add(right_frame, minsize=500)

        # Left Pane: Header and Mode Switch
        left_header = tk.Frame(left_frame)
        left_header.pack(fill=tk.X, pady=(0, 5))
        tk.Label(left_header, text="Processed Images:").pack(side=tk.LEFT)

        view_mode_var = tk.StringVar(value=self.cfg.get("batch_view_mode", "List"))

        list_container = tk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        # Listbox Mode
        listbox_frame = tk.Frame(list_container)
        list_scroll = tk.Scrollbar(listbox_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(listbox_frame, yscrollcommand=list_scroll.set, exportselection=False)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=listbox.yview)

        # Thumbnail Mode
        thumb_frame = tk.Frame(list_container)
        thumb_canvas = tk.Canvas(thumb_frame, highlightthickness=0)
        thumb_scroll = tk.Scrollbar(thumb_frame, orient="vertical", command=thumb_canvas.yview)
        thumb_inner_frame = tk.Frame(thumb_canvas)

        thumb_inner_frame.bind(
            "<Configure>",
            lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all"))
        )
        thumb_canvas.create_window((0, 0), window=thumb_inner_frame, anchor="nw", width=250)
        thumb_canvas.configure(yscrollcommand=thumb_scroll.set)
        thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        thumb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Add mousewheel scrolling safely for thumb canvas
        def _on_mousewheel(event):
            thumb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux_up(event):
            thumb_canvas.yview_scroll(-1, "units")

        def _on_mousewheel_linux_down(event):
            thumb_canvas.yview_scroll(1, "units")

        def _bind_scroll(e):
            thumb_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            thumb_canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            thumb_canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)

        def _unbind_scroll(e):
            thumb_canvas.unbind_all("<MouseWheel>")
            thumb_canvas.unbind_all("<Button-4>")
            thumb_canvas.unbind_all("<Button-5>")

        thumb_canvas.bind("<Enter>", _bind_scroll)
        thumb_canvas.bind("<Leave>", _unbind_scroll)

        self.thumbnail_images = []
        self.thumb_labels = []

        def update_view_mode(*args):
            self.cfg.set("batch_view_mode", view_mode_var.get())
            if view_mode_var.get() == "List":
                thumb_frame.pack_forget()
                listbox_frame.pack(fill=tk.BOTH, expand=True)
            else:
                listbox_frame.pack_forget()
                thumb_frame.pack(fill=tk.BOTH, expand=True)

        tk.Radiobutton(left_header, text="Preview", variable=view_mode_var, value="Preview",
                       command=update_view_mode).pack(side=tk.RIGHT)
        tk.Radiobutton(left_header, text="List", variable=view_mode_var, value="List", command=update_view_mode).pack(
            side=tk.RIGHT)

        def on_thumb_click(idx):
            self.current_preview_idx = idx
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(idx)
            on_select_sync()

        # Keyboard Navigation for "Preview" mode
        def on_up_arrow(e):
            if view_mode_var.get() == "Preview" and self.current_preview_idx > 0:
                on_thumb_click(self.current_preview_idx - 1)

        def on_down_arrow(e):
            if view_mode_var.get() == "Preview" and self.current_preview_idx < len(self.batch_results) - 1:
                on_thumb_click(self.current_preview_idx + 1)

        top.bind("<Up>", on_up_arrow)
        top.bind("<Down>", on_down_arrow)

        for i, res in enumerate(self.batch_results):
            listbox.insert(tk.END, res["path"].name)
            listbox.itemconfig(tk.END, {'fg': listbox_fg})

            img_thumb = res["result_img"].copy()
            img_thumb.thumbnail((120, 120))

            bg = Image.new("RGBA", img_thumb.size, (200, 200, 200, 255))
            for x in range(0, bg.width, 10):
                for y in range(0, bg.height, 10):
                    if (x // 10 + y // 10) % 2 == 0:
                        bg.paste((255, 255, 255, 255), [x, y, x + 10, y + 10])
            final_thumb = Image.alpha_composite(bg, img_thumb)

            tk_img = ImageTk.PhotoImage(final_thumb)
            self.thumbnail_images.append(tk_img)

            lbl = tk.Label(thumb_inner_frame, image=tk_img, text=res["path"].name, compound=tk.TOP, pady=5)
            lbl.bind("<Button-1>", lambda e, idx=i: on_thumb_click(idx))
            lbl.pack(fill=tk.X, padx=5, pady=2)
            self.thumb_labels.append(lbl)

        update_view_mode()

        # Info Panels
        info_left_frame = tk.Frame(left_frame, pady=5)
        info_left_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_info_left = tk.Label(info_left_frame, text="", justify=tk.LEFT, font=("Arial", 9))
        self.lbl_info_left.pack(anchor="w")

        info_right_frame = tk.Frame(right_frame, pady=5)
        info_right_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_info_right = tk.Label(info_right_frame, text="", justify=tk.LEFT, font=("Arial", 9))
        self.lbl_info_right.pack(anchor="w")

        # Right Pane: Controls & Canvas
        ctrl_frame = tk.Frame(right_frame, pady=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(ctrl_frame, text="Background:").pack(side=tk.LEFT, padx=(0, 10))

        preview_canvas = tk.Canvas(right_frame, bg="#2b2b2b")
        preview_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

        # Color Buttons
        def set_bg(ctype, val):
            if self.batch_results:
                self.batch_results[self.current_preview_idx]["bg_type"] = ctype
                self.batch_results[self.current_preview_idx]["bg_val"] = val
                render_preview()

        def set_bg_custom() -> None:
            color = colorchooser.askcolor(title="Choose background color")
            if color[0]:
                set_bg("color", tuple(int(c) for c in color[0]))

        def set_bg_image() -> None:
            fpath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
            if fpath:
                try:
                    img = Image.open(fpath).convert("RGBA")
                    set_bg("image", img)
                    bg_mode_combo.pack(side=tk.LEFT, padx=5)
                except Exception as e:
                    messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

        tk.Button(ctrl_frame, text="Transparent", command=lambda: set_bg("color", None)).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="White", command=lambda: set_bg("color", (255, 255, 255))).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Black", command=lambda: set_bg("color", (0, 0, 0))).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Custom Color...", command=set_bg_custom).pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="|").pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Image...", command=set_bg_image).pack(side=tk.LEFT, padx=5)

        # Image Mode Dropdown (dynamically shown)
        bg_mode_var = tk.StringVar(value="Center")
        bg_mode_combo = ttk.Combobox(ctrl_frame, textvariable=bg_mode_var, values=["Center", "Stretch"],
                                     state="readonly", width=8)

        def on_bg_mode_change(*args):
            if self.batch_results:
                self.batch_results[self.current_preview_idx]["bg_img_mode"] = bg_mode_var.get()
                render_preview()

        bg_mode_combo.bind("<<ComboboxSelected>>", on_bg_mode_change)

        if self.is_batch_mode:
            tk.Label(ctrl_frame, text="|").pack(side=tk.LEFT, padx=5)

            def apply_to_all() -> None:
                if not self.batch_results: return
                curr = self.batch_results[self.current_preview_idx]
                for res in self.batch_results:
                    res["bg_type"] = curr["bg_type"]
                    res["bg_val"] = curr["bg_val"]
                    res["bg_img_mode"] = curr["bg_img_mode"]
                messagebox.showinfo("Applied", "Background settings applied to all images in the batch.")

            tk.Button(ctrl_frame, text="Apply to All", command=apply_to_all, bg="#2196F3", fg="white").pack(
                side=tk.LEFT, padx=5)

        self.reedit_var = tk.BooleanVar(value=False)
        reedit_chk = tk.Checkbutton(ctrl_frame, text="Mark for Re-edit ✏️", variable=self.reedit_var,
                                    command=lambda: toggle_reedit(), font=("Arial", 9, "bold"))
        reedit_chk.pack(side=tk.RIGHT, padx=10)

        # Theme adjustments
        if is_dark:
            top.configure(bg="#1e1e1e")
            left_frame.configure(bg="#1e1e1e")
            left_header.configure(bg="#1e1e1e")
            thumb_inner_frame.configure(bg="#1e1e1e")
            thumb_frame.configure(bg="#1e1e1e")
            thumb_canvas.configure(bg="#1e1e1e")
            right_frame.configure(bg="#1e1e1e")
            ctrl_frame.configure(bg="#1e1e1e")
            info_left_frame.configure(bg="#1e1e1e")
            info_right_frame.configure(bg="#1e1e1e")
            for child in top.winfo_children():
                if isinstance(child, tk.Label): child.configure(bg="#1e1e1e", fg="#ffffff")
                if isinstance(child, tk.PanedWindow): child.configure(bg="#1e1e1e")
            for child in left_frame.winfo_children():
                if isinstance(child, tk.Label): child.configure(bg="#1e1e1e", fg="#ffffff")
            for child in left_header.winfo_children():
                if isinstance(child, tk.Radiobutton): child.configure(bg="#1e1e1e", fg="#ffffff", selectcolor="#2b2b2b")
                if isinstance(child, tk.Label): child.configure(bg="#1e1e1e", fg="#ffffff")
            for child in ctrl_frame.winfo_children():
                if isinstance(child, tk.Label): child.configure(bg="#1e1e1e", fg="#ffffff")
            for lbl in self.thumb_labels:
                lbl.configure(bg="#1e1e1e", fg="#ffffff")
            self.lbl_info_left.configure(bg="#1e1e1e", fg="#a0c4ff")
            self.lbl_info_right.configure(bg="#1e1e1e", fg="#a0c4ff")
            listbox.configure(bg="#2b2b2b", fg="#ffffff", selectbackground="#0078D7")
            reedit_chk.configure(bg="#1e1e1e", fg=reedit_fg, selectcolor="#2b2b2b")
        else:
            reedit_chk.configure(fg=reedit_fg)
            self.lbl_info_left.configure(fg="#005a9e")
            self.lbl_info_right.configure(fg="#005a9e")

        def composite_image(extracted_img: Image.Image, res: dict) -> Image.Image:
            """Composites the extracted image over a background (color or image).
            
            Args:
                extracted_img: RGBA PIL Image with extracted object (transparent outside mask).
                res: Result dictionary with keys:
                    'bg_type': 'color' or 'image'
                    'bg_val': RGB tuple (r, g, b) if color, None for transparent checkerboard,
                        or PIL Image if bg_type='image'
                    'bg_img_mode': 'Center' (center image, pad) or 'Stretch' (scale to fill)
            
            Returns:
                Composited image showing extracted object on selected background.
            """
            bg_type = res["bg_type"]
            bg_val = res["bg_val"]

            if bg_type == "color":
                if bg_val is None:
                    bg = Image.new("RGBA", extracted_img.size, (200, 200, 200, 255))
                    for x in range(0, bg.width, 20):
                        for y in range(0, bg.height, 20):
                            if (x // 20 + y // 20) % 2 == 0:
                                bg.paste((255, 255, 255, 255), [x, y, x + 20, y + 20])
                    return Image.alpha_composite(bg, extracted_img)
                else:
                    bg = Image.new("RGBA", extracted_img.size, bg_val + (255,))
                    return Image.alpha_composite(bg, extracted_img)

            elif bg_type == "image":
                bg_img = bg_val.copy()
                bg_mode = res.get("bg_img_mode", "Center")

                if bg_mode == "Stretch":
                    bg_img = bg_img.resize(extracted_img.size, Image.Resampling.LANCZOS)
                elif bg_mode == "Center":
                    temp = Image.new("RGBA", extracted_img.size, (0, 0, 0, 0))
                    # Center the image
                    x = (extracted_img.width - bg_img.width) // 2
                    y = (extracted_img.height - bg_img.height) // 2
                    temp.paste(bg_img, (x, y))
                    bg_img = temp

                return Image.alpha_composite(bg_img, extracted_img)

        def render_preview() -> None:
            """Renders the current selected image with applied background settings to the preview canvas."""
            if not self.batch_results: return

            current_res = self.batch_results[self.current_preview_idx]
            extracted_img = current_res["result_img"]

            # Show/Hide BG mode dropdown
            if current_res["bg_type"] == "image":
                bg_mode_var.set(current_res.get("bg_img_mode", "Center"))
                bg_mode_combo.pack(side=tk.LEFT, padx=5)
            else:
                bg_mode_combo.pack_forget()

            final_preview = composite_image(extracted_img, current_res)

            canvas_w = preview_canvas.winfo_width()
            canvas_h = preview_canvas.winfo_height()
            if canvas_w < 10: canvas_w, canvas_h = 600, 500

            scale = min(canvas_w / final_preview.width, canvas_h / final_preview.height)
            if scale < 1.0:
                new_w, new_h = int(final_preview.width * scale), int(final_preview.height * scale)
                final_preview = final_preview.resize((new_w, new_h), Image.Resampling.LANCZOS)

            tk_preview = ImageTk.PhotoImage(final_preview)
            preview_canvas.delete("all")
            preview_canvas.image = tk_preview
            preview_canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=tk_preview)

            for i, lbl in enumerate(self.thumb_labels):
                if i == self.current_preview_idx:
                    lbl.configure(bg="#0078D7" if is_dark else "#a0c4ff")
                else:
                    lbl.configure(bg="#1e1e1e" if is_dark else "#f0f0f0")

            # Update Process Information Text
            old_name = current_res["path"].name
            new_name = self._get_output_filename(current_res)
            w, h = extracted_img.size
            self.lbl_info_left.config(text=f"File: {old_name}\n └─> {new_name}")
            self.lbl_info_right.config(
                text=f"Resolution: {w}x{h} px\nModel: {current_res['engine']}\nHardware: {current_res['device']}")

        def on_select_sync():
            """Synchronizes the re-edit checkbox and re-renders preview for the selected image."""
            self.reedit_var.set(self.batch_results[self.current_preview_idx].get("reedit", False))
            render_preview()

        def on_select(event):
            """Handles listbox selection events.
            
            Args:
                event: The listbox selection event.
            """
            selection = listbox.curselection()
            if selection:
                self.current_preview_idx = selection[0]
                on_select_sync()

        listbox.bind('<<ListboxSelect>>', on_select)
        preview_canvas.bind("<Configure>", lambda e: render_preview())

        def toggle_reedit() -> None:
            """Toggles the re-edit flag for the current image and updates list display."""
            if not self.batch_results: return
            idx = self.current_preview_idx
            is_reedit = self.reedit_var.get()
            self.batch_results[idx]["reedit"] = is_reedit

            name = self.batch_results[idx]["path"].name
            display_name = f"[RE-EDIT] {name}" if is_reedit else name

            listbox.delete(idx)
            listbox.insert(idx, display_name)
            listbox.selection_set(idx)
            listbox.itemconfig(idx, {'fg': reedit_fg if is_reedit else listbox_fg})
            self.thumb_labels[idx].configure(text=display_name, fg=reedit_fg if is_reedit else listbox_fg)

        btn_frame = tk.Frame(top)
        if top.cget('bg') == "#1e1e1e": btn_frame.configure(bg="#1e1e1e")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15, padx=20)

        def clean_up_progress():
            """Resets progress tracking and clears batch task/result queues."""
            self.overall_progress_var.set(0)
            self.item_progress_var.set(0)
            self.batch_tasks = []
            self.batch_results = []

        def approve_all() -> None:
            """Saves approved images to disk with configured backgrounds, marks re-edit items for reprocessing."""
            saved_count = 0
            reedit_files = []

            for res in self.batch_results:
                if res.get("reedit"):
                    reedit_files.append(res["path"])
                    continue

                extracted_img = res["result_img"]
                bg_type = res["bg_type"]
                bg_val = res["bg_val"]
                path = res["path"]

                out_dir = path.parent / "masked_photos"
                out_dir.mkdir(exist_ok=True)

                final_name = self._get_output_filename(res)
                save_path = out_dir / final_name

                if bg_type == "color" and bg_val is None:
                    extracted_img.save(save_path)
                else:
                    # Reuse compositing logic but skip the checkerboard for pure output
                    if bg_type == "color":
                        bg = Image.new("RGBA", extracted_img.size, bg_val + (255,))
                        Image.alpha_composite(bg, extracted_img).convert("RGB").save(save_path, quality=95)
                    elif bg_type == "image":
                        bg_img = bg_val.copy()
                        bg_mode = res.get("bg_img_mode", "Center")

                        if bg_mode == "Stretch":
                            bg_img = bg_img.resize(extracted_img.size, Image.Resampling.LANCZOS)
                        elif bg_mode == "Center":
                            temp = Image.new("RGBA", extracted_img.size, (0, 0, 0, 0))
                            x = (extracted_img.width - bg_img.width) // 2
                            y = (extracted_img.height - bg_img.height) // 2
                            temp.paste(bg_img, (x, y))
                            bg_img = temp

                        Image.alpha_composite(bg_img, extracted_img).convert("RGB").save(save_path, quality=95)

                saved_count += 1

            msg = f"Saved {saved_count} images."
            if reedit_files:
                msg += f" {len(reedit_files)} marked for re-edit."
            self.update_status_text(msg)

            clean_up_progress()
            top.destroy()

            if reedit_files:
                self.is_batch_mode = True
                self.batch_files = reedit_files
                self.load_next_batch_image()
            else:
                self.is_batch_mode = False

        def discard() -> None:
            """Closes the approval window and clears all batch results without saving."""
            self.update_status_text("Review window closed / discarded.")
            clean_up_progress()
            self.is_batch_mode = False
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", discard)

        tk.Button(btn_frame, text="✅ Process & Save", command=approve_all, bg="#4CAF50", fg="white",
                  font=("Arial", 12, "bold"),
                  width=20).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ Discard All", command=discard, bg="#f44336", fg="white",
                  font=("Arial", 12, "bold"),
                  width=15).pack(side=tk.RIGHT, padx=10)

        if self.batch_results:
            listbox.selection_set(0)
            on_thumb_click(0)