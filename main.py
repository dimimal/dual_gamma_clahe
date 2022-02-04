import os
import argparse
from skimage import io
from dg_clahe import dual_gamma_clahe
import matplotlib.pyplot as plt


def parse_kernel(input):
    if (isinstance(input, list) and len(input) == 2) or isinstance(input, int):
        raise ValueError(f"kernel should be either int or a sequence of 2 ints but got {input}")

def file_exists(file_):
    if not os.path.exists(file_):
        raise ValueError(f"Image file not found {file_}")

def check_folder(folder):
    if os.path.isfile(folder):
        raise ValueError(f'Output directory is an existing file {folder}, please set a folder here or leave it default.')
    elif not os.path.isdir(folder):
        os.makedirs(folder)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Size of the kernel, if int then its (int, int)", type=str)
    parser.add_argument("--kernel", help="Size of the kernel, if int then its (height, width)", nargs="+", type=parse_kernel, default=32)
    parser.add_argument("--alpha", help="Alpha parameter of the algorithm", type=float, default=40)
    parser.add_argument("--delta", help="The Delta threshold of the algorithm", type=int, default=50)
    parser.add_argument("--p", help="The factor for the computation of clip limits", type=int, default=1.5)
    parser.add_argument("--show", help="Display the 2 figures with matplotlib, before and after equalization", action='store_true')
    parser.add_argument("--out", help="Output directory of the equalized image. Default folder is the ./images folder", type=check_folder)
    return parser.parse_args()


def main(args):
    image_name = 'equalized_'+ args.image.split(os.sep)[-1]
    image = io.imread(args.image)
    equalized_image = dual_gamma_clahe(image.copy(), block_size=args.kernel, alpha=args.alpha, delta=args.delta, pi=args.p, bins=256)

    if args.show:
        if image.ndim == 2:
            cmap = 'gray'
        else:
            cmap = None
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image, cmap=cmap)
        ax[1].imshow(equalized_image, cmap=cmap)
        ax[0].set_title('Input Image')
        ax[1].set_title('Equalized Image with')
        plt.show()

    # Store image
    io.imsave(os.path.join(args.out, image_name))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)