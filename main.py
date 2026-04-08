from rembg import remove, new_session
from pathlib import Path


class OP:
    session = None
    in_dir = None
    out_dir = None
    num_total = 0
    num_done = 0


def check_input_data(input_dir: str = "./raw_photos") -> bool:
    """
        Checks if the input files are valid.
       :param input_dir: Input directory for photos, default is "./raw_photos"
    """
    OP.in_dir = Path(input_dir)
    if OP.in_dir.exists():
        for file in OP.in_dir.glob("*.jpg"):
            OP.num_total += 1
        print(f"Found {OP.num_total} images.")
        # TODO: Add proper logic to verify that at least 1 file exists and is a valid supported format.
        return True
    print("There are no input files, directory will be created.")
    OP.in_dir.mkdir(parents=True, exist_ok=True)
    return False


def make_output_dir(output_dir: str = "./masked_photos", clear: bool = False) -> None:
    """
        Creates the ObjectPresenter output directory for photo processing.

        :param output_dir: Output directory for photos, default is "./masked_photos"
        :param clear: Clear output directory if True, default is False
    """
    OP.out_dir = Path(output_dir)
    if clear:
        # We delete the files inside the directory
        for file in OP.out_dir.glob("*"):
            file.unlink()
        OP.out_dir.rmdir()
    OP.out_dir.mkdir(parents=True, exist_ok=True)


def removebg() -> None:
    """
    Processes images by removing the background and saving them as transparent PNGs.
    """
    for img_file in OP.in_dir.glob("*.jpg"):
        print(f"Processing: {img_file.name}")
        with open(img_file, "rb") as file:
            input_image_data = file.read()
        output_image_data = remove(input_image_data, session=OP.session)
        save_path = OP.out_dir / f"{img_file.stem}_masked.png"
        with open(save_path, "wb") as file:
            file.write(output_image_data)
        OP.num_done += 1
        print(f"Progress: {OP.num_done}/{OP.num_total} ({round(OP.num_done/OP.num_total*100,2)}%)")


if __name__ == "__main__":
    print("Starting...")
    try:
        OP.session = new_session()
    except ValueError:
        print("Could not create new session")
        exit(1)
    print("Session created:", OP.session.model_name)
    if not check_input_data():
        exit(2)
    make_output_dir(clear=False)
    removebg()
    print("Done!")
