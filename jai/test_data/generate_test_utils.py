import argparse
from pathlib import Path

from jai.utilities import read_image_folder


def generate_read_image_folder(image_folder):
    img_data = read_image_folder(image_folder=image_folder,
                                 id_pattern="img(\d+)")
    print()
    print(img_data)
    img_data.to_pickle(image_folder / "dataframe_img.pkl", protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p",
                        "--path",
                        default="test_imgs",
                        help="Image paths.")
    args = parser.parse_args()

    generate_read_image_folder(Path(args.path))
