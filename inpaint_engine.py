from PIL import Image
import numpy as np


class InpaintEngine:
    """Module for inpainting. Supports multiple underlying engines (OpenCV, LaMa)."""

    def __init__(self) -> None:
        self.lama_model = None

    def _load_lama(self) -> None:
        """Lazy loads the LaMa model."""
        if self.lama_model is None:
            # Requires: pip install simple-lama
            from simple_lama_inpainting import SimpleLama
            self.lama_model = SimpleLama()

    def _apply_opencv(self, img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Uses classic OpenCV Telea algorithm for inpainting."""
        import cv2

        # Convert PIL -> OpenCV (RGB to BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
        # Mask must be 8-bit, 1-channel
        mask_cv = np.array(mask_pil.convert('L'))

        # Telea algorithm (fast, good for small scratches, okayish for thumbs)
        inpainted = cv2.inpaint(img_cv, mask_cv, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Back to PIL
        return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGBA))

    def _apply_lama(self, img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Uses LaMa neural network for inpainting."""
        self._load_lama()
        # simple_lama takes PIL objects directly
        # Needs RGB image and Grayscale (L) mask
        result = self.lama_model(img_pil.convert('RGB'), mask_pil.convert('L'))
        # FIX: simple-lama sometimes pads/resizes images under the hood to meet
        # tensor size constraints (multiples of 8). We MUST force it back to the
        # exact input dimensions to avoid 'ValueError: images do not match' when applying alpha.
        if result.size != img_pil.size:
            result = result.resize(img_pil.size, Image.Resampling.LANCZOS)
        return result.convert('RGBA')

    def process(self, img_pil: Image.Image, mask_pil: Image.Image, method: str) -> Image.Image:
        """Main API for the module. Delegates to the selected engine."""
        if method == "OpenCV":
            return self._apply_opencv(img_pil, mask_pil)
        elif method == "LaMa":
            return self._apply_lama(img_pil, mask_pil)
        else:
            raise ValueError(f"Unknown inpainting method: {method}")