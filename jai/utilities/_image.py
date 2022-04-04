import base64
import pandas as pd

from pathlib import Path
from PIL import Image
from typing import List, Tuple
from tqdm import tqdm
from io import BytesIO

__all__ = ["read_image_folder"]


def read_image_folder(image_folder: str = None,
                      images: List = None,
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
    images : List of Path, optional
        List of Paths for each image. The default is None.
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
    name = "image_base64"
    if image_folder is not None:
        name = Path(image_folder).name
        images = Path(image_folder).iterdir()
    elif images is None:
        raise ValueError(
            "must pass the folder of the images or a list with the paths of each image."
        )

    ids = []
    encoded_images = []
    filenames = []
    corrupted_files = []
    for i, filename in enumerate(tqdm(images)):
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
                im_file = BytesIO()
                img.save(im_file, format="PNG")
                encoded_string = base64.b64encode(
                    im_file.getvalue()).decode("utf-8")

                # test if decoding is working
                Image.open(BytesIO(
                    base64.b64decode(encoded_string))).convert("RGB")

                encoded_images.append(encoded_string)
                filenames.append(filename.name)
                ids.append(i)
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

    index = pd.Index(ids, name='id')
    return pd.DataFrame({
        name: encoded_images,
        "filename": filenames
    },
                        index=index)
