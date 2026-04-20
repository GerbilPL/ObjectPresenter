# ObjectPresenter
Takes the item you took a photo of, in-fills it and removes the background.

# Features

- Background removal based on user selection area
- TK gui for easier usage
- Exporting the processed images with custom background color.

# Planned

- In-painting of missing areas. Preferably using other images as base, but the current goal is to hallucinate something close
- Downloader for SAM model checkpoints (Low priority)
- Revamping the UI (Low priority)
- Adding support for web servers. (Low priority)
- Adding the ability to process videos and extract key frames.
- Other export background options. (Low priority)

# Requirements

The project currently uses rembg and Segment Anything Model for removing the background layer. The goal for release version is to remove the rembg dependency and add a downloader for SAM.