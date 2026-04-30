# ObjectPresenter
Takes the item you took a photo of, in-fills it and removes the background.

## Features

- Background removal based on user selection area
- TK gui for easier usage
- Exporting the processed images with custom background.
- Batch processing (you still need to select on each image as of now)
- 

## Experimental (or close to working)
- In-painting of missing areas. Preferably using other images as base, but the current goal is to hallucinate something close

## Planned

- Downloader for SAM model checkpoints (Low priority)
- Adding support for web servers. (Low priority)
- Adding the ability to process videos and extract key frames.

## Requirements

The project currently uses rembg and Segment Anything Model for removing the background layer. ~~The goal for release version is to remove the rembg dependency and add a downloader for SAM.~~
> [!NOTE]
> **After thurther testing, rembg model is still going strong in some places, so it will stick for now**

## Requirements:

While you can only use SAM or rembg, if you can spare some space, get both.
Similarly with LaMa and OpenCV, but LaMa is recommended.
- [segment anything](https://github.com/facebookresearch/segment-anything)
- [rembg](https://github.com/danielgatis/rembg)
  - `rembg[cpu]` or `rembg[gpu]`<sup>1</sup>
- [torch and torchvision](https://pytorch.org/)<sup>2</sup>
- [opencv](https://github.com/opencv/opencv-python)
- [simple lama inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- [numpy](https://numpy.org/)
- [pillow](https://python-pillow.github.io/)

And oh well you might want to use the gui, which uses Tkinter.

> [!NOTE]
> If pip or other package manager complains about incompatible versions try to install the ones supported by the models, since they might be more fussy. Other things should work fine.

> **1** Runtime for rembg

> **2** Make sure to get the correct version, depending on your system and usage
