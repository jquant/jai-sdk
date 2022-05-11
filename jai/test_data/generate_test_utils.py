from pathlib import Path

from jai.utilities import read_image_folder


def generate_read_image_folder(image_folder=Path("test_imgs")):
    img_data = read_image_folder(image_folder=image_folder,
                                 id_pattern="img(\d+)")
    print()
    print(img_data)
    img_data.to_pickle(Path("test_imgs/dataframe_img.pkl"))


if __name__ == '__main__':
    generate_read_image_folder()
