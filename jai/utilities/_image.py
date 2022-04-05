import pandas as pd

from base64 import b64encode, b64decode
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Union
from tqdm import tqdm
from io import BytesIO

__all__ = ["read_image_folder"]


def encode_image(image) -> str:
    img_file = BytesIO()
    image.save(img_file, format="PNG")
    return b64encode(img_file.getvalue()).decode("utf-8")


def decode_image(encoded_string):
    return Image.open(BytesIO(b64decode(encoded_string))).convert("RGB")


def read_image_folder(image_folder: Union[str, Path],
                      resize: Tuple[int, int] = None,
                      ignore_corrupt=False,
                      extensions: List = [".png", ".jpg", ".jpeg"]):
    """
    Function to read images from folder and transform to a format compatible to jai.

    Must pass the folder of the images or a list of paths for each image.

    Parameters
    ----------
    image_folder : str or Path, optional
        Path for the images folder. The default is None.
    new_size : Tuple of int, optional
        New shape to resize images. The default is None.
    ignore_corrupt : TYPE, optional
        If ignore corrupt images. If True, could probably result in a internal
        error later on. The default is False.
    extensions : List, optional
        List of acceptable extensions. The default is [".png", ".jpg", ".jpeg"].

    Raises
    ------
    ValueError
        If not passed a image folder or list of images.

    Returns
    -------
    pd.Series
        Pandas Series with acceptable format for jai usage.

    """
    name = Path(image_folder).name

    encoded_images = []
    corrupted_files = []
    for i, filename in enumerate(tqdm(Path(image_folder).iterdir())):
        if filename.suffix in extensions:
            try:
                im = Image.open(filename)
                im.verify(
                )  # I perform also verify, don't know if he sees other types o defects
                im.close()  # reload is necessary in my case
                im = Image.open(filename)
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()

                img = Image.open(filename)
                if resize is not None:
                    img = img.resize(resize, Image.ANTIALIAS).convert('RGB')
                encoded_string = encode_image(img)

                # test if decoding is working
                decode_image(encoded_string)

                encoded_images.append({'id':i, name:encoded_string, "filename": filename.name})
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                if ignore_corrupt:
                    corrupted_files.append(filename)
                    continue
                raise ValueError(f"file {filename} seems to be corrupted.")

    if len(corrupted_files) > 0:
        print("Here are the files that seem to be corrupted:")
        [print(f"{f}") for f in corrupted_files]

    return pd.DataFrame(encoded_images).set_index("id")
