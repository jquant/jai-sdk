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
                      handle_errors: str = "ignore",
                      extensions: List = [".png", ".jpg", ".jpeg"]):
    """
    Function to read images from folder and transform to a format compatible
    to jai.

    Must pass the folder of the images or a list of paths for each image.

    Parameters
    ----------
    image_folder : str or Path, optional
        Path for the images folder. The default is None.
    new_size : Tuple of int, optional
        New shape to resize images. The default is None.
    handle_errors : str, optional
        Whether to ignore errors and skipped files. 
        If "ignore", could probably result in a internal error later on.
        The default is "ignore".
    extensions : List, optional
        List of acceptable extensions.
        The default is [".png", ".jpg", ".jpeg"].

    Raises
    ------
    ValueError
        If not passed a image folder or list of images.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe with acceptable format for jai usage.

    """
    image_folder = Path(image_folder)
    name = image_folder.name

    if handle_errors not in ['raise', 'warn', 'ignore']:
        raise ValueError("handle_errors must be 'raise', 'warn' or 'ignore'")

    encoded_images = []
    corrupted_files = []
    ignored_files = []
    for i, filename in enumerate(tqdm(image_folder.iterdir())):
        if filename.suffix in extensions:
            try:
                im = Image.open(filename)
                im.verify()
                im.close()
                im = Image.open(filename)
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()

                img = Image.open(filename)
                if resize is not None:
                    img = img.resize(resize, Image.ANTIALIAS).convert('RGB')
                encoded_string = encode_image(img)

                # test if decoding is working
                decode_image(encoded_string)

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as error:
                if handle_errors == 'raise':
                    raise ValueError(
                        f"file {filename} seems to be corrupted. {error}")

                corrupted_files.append(filename)

            encoded_images.append({
                'id': i,
                name: encoded_string,
                "filename": filename.name
            })
        else:
            if handle_errors == 'raise':
                raise ValueError(
                    f"file {filename} does not have the proper extension.\
                        acceptable extensions: {extensions}")
            ignored_files.append(filename)

    if handle_errors == 'warn' and len(ignored_files) > 0:
        print("Here are the ignored files:")
        ignored_message = '\n'.join(ignored_files)
        print(f"{ignored_message}")

    if handle_errors == 'warn' and len(corrupted_files) > 0:
        print("Here are the files that seem to be corrupted:")
        corrupted_message = '\n'.join(corrupted_files)
        print(f"{corrupted_message}")

    return pd.DataFrame(encoded_images).set_index("id")
