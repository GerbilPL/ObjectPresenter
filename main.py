import tkinter as tk
import argparse
import sys
from gui import ObjectPickerApp


def check_cuda_cli(device_preference):
    """Silent check for CLI users regarding hardware acceleration."""
    if device_preference == "CPU":
        return
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n!!! WARNING: CUDA (GPU) not detected. PyTorch will use the CPU.")
            print("   Processing will be significantly slower.\n")
            if device_preference == "CUDA":
                print("!!!  ERROR: CUDA was forced (--device CUDA) but is unavailable.")
                sys.exit(1)
    except ImportError:
        print("\n!!! WARNING: PyTorch is not installed. SAM engine will fail.\n")


def main():
    """Application entry point with optional CLI support."""
    parser = argparse.ArgumentParser(description="Object Picker - Background Removal & Inpainting")
    parser.add_argument('--cli', action='store_true', help='Run in console mode (no GUI)')
    parser.add_argument('--input', type=str, help='Path to input image (for CLI mode)')
    parser.add_argument('--batch-dir', type=str, help='Path to directory for batch processing')
    parser.add_argument('--device', type=str, choices=['Auto', 'CUDA', 'CPU'], default='Auto',
                        help='Force Hardware execution provider')

    args = parser.parse_args()

    if args.cli:
        print("Starting in CLI mode...")
        check_cuda_cli(args.device)
        print("Batch processing and CLI module is under construction.")
        # DEV NOTE: The ProgressTracker is now decoupled and ready to be used here
        # to iterate over 'batch-dir' entirely independently of Tkinter.
        sys.exit(0)

    # Start standard GUI mode
    root = tk.Tk()
    app = ObjectPickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()