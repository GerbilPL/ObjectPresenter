import os
import numpy as np
from PIL import Image
from typing import Tuple

from inpaint_engine import InpaintEngine
from progress_tracker import ProgressTracker


class ImageProcessor:
    """Handles all heavy lifting regarding image manipulation, AI model inference, and masking."""

    def __init__(self):
        """Initializes ImageProcessor with lazy-loaded model placeholders."""
        self.rembg_session = None
        self.sam_predictor = None
        self.inpaint_model = InpaintEngine()

    def clear_models(self) -> None:
        """
        Forcefully unloads currently loaded models and clears GPU VRAM.
        Used to seamlessly apply device changes (CPU <-> CUDA) without restarting.
        """
        self.rembg_session = None

        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        print("DEV LOG: Models cleared from memory. Ready for re-initialization.")

    def preload_inpaint(self, method: str) -> None:
        """Preloads inpainting module to catch import errors before processing starts.
        
        Attempts to import and initialize the selected inpainting module.
        If import fails, exception is raised early so user can fix dependencies.
        
        Args:
            method: Inpainting method to load: 'OpenCV' (requires cv2) or
                'LaMa' (requires simple-lama package).
        
        Raises:
            ImportError: If required package not installed.
        """
        if method == "OpenCV":
            import cv2  # Trigger import error if not installed
        elif method == "LaMa":
            self.inpaint_model._load_lama()

    def load_rembg(self, device_preference: str = "Auto") -> None:
        """Lazy loads the rembg background removal model on first call.
        
        Initializes rembg's isnet-general-use model with configured execution
        providers. Subsequent calls are no-ops (model already cached).
        
        Args:
            device_preference: Where to run model: 'Auto' (auto-detect, prefer GPU),
                'CUDA' (force ONNX CUDA provider), 'CPU' (force ONNX CPU provider).
                Default is 'Auto'.
        
        Raises:
            Exception: If rembg package missing or model download fails.
        """
        if self.rembg_session is None:
            from rembg import new_session
            providers = None  # None lets rembg use its default Auto behavior
            if device_preference == "CPU":
                providers = ["CPUExecutionProvider"]
            elif device_preference == "CUDA":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self.rembg_session = new_session("isnet-general-use", providers=providers)

    def load_sam(self, device_preference: str = "Auto") -> None:
        """Lazy loads the SAM (Segment Anything Model) checkpoint on first call.
        
        Loads the Vision Transformer B (vit_b) variant to selected device.
        Subsequent calls are no-ops (model already cached in self.sam_predictor).
        
        Args:
            device_preference: Where to run model: 'Auto' (auto-detect CUDA availability),
                'CUDA' (force CUDA, error if unavailable), 'CPU' (force CPU). Default 'Auto'.
        
        Raises:
            FileNotFoundError: If sam_vit_b_01ec64.pth checkpoint missing.
            RuntimeError: If device_preference='CUDA' but CUDA unavailable.
            Exception: If PyTorch or segment_anything package missing.
        """
        if self.sam_predictor is None:
            checkpoint_path = "sam_vit_b_01ec64.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found: '{checkpoint_path}'")

            import torch
            from segment_anything import sam_model_registry, SamPredictor

            device = "cpu"
            if device_preference in ["Auto", "CUDA"]:
                if torch.cuda.is_available():
                    device = "cuda"
                elif device_preference == "CUDA":
                    raise RuntimeError(
                        "CUDA was requested but is not available. Please check your PyTorch installation.")

            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)

    def process(
            self,
            original_img: Image.Image,
            working_bbox: Tuple[int, int, int, int],
            engine: str,
            inpaint_enabled: bool,
            inpaint_method: str,
            device_preference: str = "Auto",
            tracker: ProgressTracker = None
    ) -> Image.Image:
        """Main pipeline for extracting an object and optionally inpainting background.

        Executes two-phase processing: (1) Background removal/segmentation using
        selected engine, (2) Optional inpainting of the mask hole region.
        
        Args:
            original_img: Full source image (RGBA format) containing object to extract.
            working_bbox: Tuple of (x1, y1, x2, y2) defining region within original_img.
                Can include margin around actual object for better edge detection.
            engine: AI engine for segmentation: 'rembg (isnet)' for fast removal,
                'SAM (vit_b)' for more precise box-guided segmentation.
            inpaint_enabled: If True, applies inpainting to fill background holes
                detected in the mask. If False, outputs extracted object as-is with alpha.
            inpaint_method: When inpaint_enabled=True, selects fill algorithm:
                'OpenCV' uses Telea algorithm (fast, good for small artifacts).
                'LaMa' uses neural network (slower, better quality for large holes).
            device_preference: Hardware to use: 'Auto' (auto-detect), 'CUDA' (force GPU),
                'CPU' (force CPU). Affects both extraction and inpainting speed.
            tracker: Optional ProgressTracker for progress reporting and cancellation.
                If provided, will be updated with progress and monitored for cancel signal.
        
        Returns:
            RGBA image with extracted object (transparency outside mask).
            When inpaint_enabled=True, background holes are filled.
        
        Raises:
            FileNotFoundError: If SAM checkpoint not found.
            RuntimeError: If CUDA forced but not available.
            ValueError: If engine or inpaint_method values invalid.
        """

        def set_prog(prog: float, msg: str):
            if tracker:
                tracker.update_progress(prog, msg)

        def check_cancel():
            if tracker:
                tracker.check_cancelled()

        output_img = None
        mask_img = 0

        set_prog(10.0, "Starting extraction setup...")
        check_cancel()

        # --- 1. Extraction Phase ---
        if engine == "rembg (isnet)":
            set_prog(20.0, "Loading rembg model... please wait.")
            self.load_rembg(device_preference)
            check_cancel()

            set_prog(40.0, "Removing background (rembg)...")
            cropped_img = original_img.crop(working_bbox)
            from rembg import remove
            output_img = remove(cropped_img, session=self.rembg_session)

            if output_img and output_img.mode == "RGBA":
                mask_img = output_img.split()[3].convert("L")
            else:
                raise ValueError("rembg failed to output RGBA image.")

        elif engine == "SAM (vit_b)":
            set_prog(20.0, "Loading SAM model... please wait.")
            self.load_sam(device_preference)
            check_cancel()

            set_prog(35.0, "Preparing image arrays for SAM...")
            image_array = np.array(original_img.convert("RGB"))
            self.sam_predictor.set_image(image_array)

            check_cancel()
            set_prog(50.0, "Generating segmentation mask...")
            input_box = np.array(working_bbox)

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None, point_labels=None,
                box=input_box[None, :], multimask_output=False,
            )

            mask_array = (masks[0] * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_array).convert("L")

            output_img = original_img.copy()
            output_img.putalpha(mask_img)
            output_img = output_img.crop(working_bbox)
            mask_img = mask_img.crop(working_bbox)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        check_cancel()

        # --- 2. Inpainting Phase ---
        if inpaint_enabled and not isinstance(mask_img, int):
            set_prog(60.0, f"Setting up inpainting ({inpaint_method})...")

            import cv2

            mask_cv = np.array(mask_img)
            _, binary = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_mask_cv = np.zeros_like(binary)

            check_cancel()
            set_prog(65.0, "Calculating hole morphology...")

            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(main_contour)
                cv2.drawContours(filled_mask_cv, [hull], -1, 255, thickness=cv2.FILLED)

            hole_mask_cv = cv2.bitwise_and(filled_mask_cv, cv2.bitwise_not(binary))
            kernel = np.ones((9, 9), np.uint8)
            hole_mask_cv_dilated = cv2.dilate(hole_mask_cv, kernel, iterations=1)

            inpaint_mask = Image.fromarray(hole_mask_cv_dilated)
            new_alpha_mask_cv = cv2.bitwise_or(binary, hole_mask_cv_dilated)
            new_alpha_mask = Image.fromarray(new_alpha_mask_cv)

            check_cancel()
            set_prog(75.0, f"Filling background using {inpaint_method}...")

            inpainted_img = self.inpaint_model.process(output_img, inpaint_mask, inpaint_method)

            check_cancel()
            set_prog(95.0, "Applying alpha masks...")
            inpainted_img.putalpha(new_alpha_mask)
            output_img = inpainted_img

        set_prog(100.0, "Processing complete!")
        return output_img