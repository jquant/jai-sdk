# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:55:28 2021

@author: Kazu
"""
import base64
import pandas as pd

from pathlib import Path
from PIL import Image
from typing import List
from tqdm import tqdm

__all__ = ["read_image_folder"]

def read_image_folder(image_folder: str = None,
                      images: List = None,
                      ignore_corrupt=False,
                      extensions=["*.png", "*.jpg", "*.jpeg"]):

    if image_folder is not None:
        images = Path(image_folder).iterdir()
    elif images is not None:
        pass
    else:
        raise ValueError(
            "must pass the folder of the images or a list with the paths of each image."
        )

    temp_img = []
    ids = []
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
                with open(filename, "rb") as image_file:
                    encoded_string = base64.b64encode(
                        image_file.read()).decode("utf-8")
                temp_img.append(encoded_string)
                ids.append(i)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                if ignore_corrupt:
                    corrupted_files.append(filename)
                    continue
                else:
                    raise ValueError(f"file {filename} seems to be corrupted.")
    if len(corrupted_files) > 0:
        print("Here are the files that seem to be corrupted:")
        [print(f"{f}") for f in corrupted_files]
    index = pd.Index(ids, name='id')
    return pd.Series(temp_img, index=index, name='image_base64')