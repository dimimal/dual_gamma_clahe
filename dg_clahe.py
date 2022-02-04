import numpy as np
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt


def compute_clip_limit(block: np.ndarray, alpha: float = 40, pi: float = 1.5, R: int = 255) -> int:
    """This function computes the beta clip limits of each image block

    Args:
        block (np.ndarray): The block of each image
        alpha (float): The alpha variance enhance parameter
        pi (float): The P factor enhanement
        R (int): The color range of the image

    Returns:
        (int): The beta clip limit for the histogram blocks
    """
    avg = block.mean()
    l_max = block.max()
    n = l_max - block.min()
    sigma = block.std()
    m = block.size
    return int((m / n) * (1 + (pi * l_max / R) + (alpha * sigma / 100.) / (avg + 0.001)))


def clip_and_redistribute_hist(hist: np.ndarray, beta: int) -> np.ndarray:
    """Clips and redistribute the excess values of the histograms among the bins using the block clip value beta

    Args:
        hist (np.ndarray): The block's histogram
        beta (int): The clipping value

    Returns:
        hist (np.ndarray): The redistributed histogram of shape (n_height_blocks, n_width_blocks, bins)
    """
    # compute the exceeded pixels mask
    mask = hist > beta
    exceed_values = hist[mask]
    bin_value = (exceed_values.sum() - exceed_values.size * beta) // hist.size
    hist[mask] = beta
    hist += bin_value
    return hist


def compute_gamma(l_max: int, d_range: int, weighted_cdf: np.ndarray) -> np.ndarray:
    """Computes the Gamma(l) function from paper in a vectorized way across the range of the color space:
    https://www.researcher-app.com/paper/708253

    Args:
        l_max (int): The global max value of the image
        d_range (int): The range of the colors in the image
        weighted_cdf (np.ndarray): The normalized CDF_w()

    Returns:
        (np.ndarray): The gamma vector of size nbins
    """
    return l_max * np.power(np.arange(d_range + 1) / l_max, (1 + weighted_cdf) / 2.)


def compute_w_en(l_alpha: int, l_max: int, cdf: np.ndarray) -> np.ndarray:
    """Computes the W_en function according to the paper in a vectorized way, for the entire cdf:
        https://www.researcher-app.com/paper/708253I

    Args:
        l_alpha (float): The global l_a variable of the 75th percentile according to the CDF
        l_max (int): the global max intensity value
        cdf (np.ndarray): The normalized cdf

    Returns:
        (np.ndarray): The W_en factors of range of the bin size
    """
    return np.power(l_max / l_alpha, 1 - (np.log(np.e + cdf) / 8))


def compute_block_coords(center_block: int, block_index: int, index: int, n_blocks: int) -> tuple:
    """This function computes the neighbour block coordinates to interpolate from the histogram. This computes the coords
    of the interpolating blocks from both axis (x, y)

    Args:
        center_block (int): The index of the center block along the pixel coords
        block_index (int): The index of the block in histogram coords
        index (int): The current index of pixel along axis
        n_blocks (int): The amount of blocks along axis

    Returns:
        (int, int): The indexes of the surrounding blocks for each axis
    """
    if index < center_block:
        block_min = block_index - 1
        if block_min < 0:
            block_min = 0
        block_max = block_index
    else:
        block_min = block_index
        block_max = block_index + 1
        if block_max >= n_blocks:
            block_max = block_index
    return block_min, block_max


def compute_mn_factors(coords: tuple, index: int) -> float:
    """Function which computes the interpolation factors of m, n according to the paper. Depending on the axis, it returns
    the proper values
        https://www.researcher-app.com/paper/708253

    Args:
        coords (tuple): The block coordinates (x, y)
        index (int): The pixel index

    Returns:
        (float): The m or n interpolation factors
    """
    if coords[1] - coords[0] == 0:
        return 0
    else:
        return (coords[1] - index) / (coords[1] - coords[0])


def dual_gamma_clahe(image: np.ndarray, block_size=(32, 32), alpha=20, pi=1.5, delta=50, bins=256):
    """The dual gamma clahe algorithm taking as inputs the image in numpy format along with the parameters. The algorithm transforms
    the image in HSV if its RGB and applies the equalization on the V channel. If grayscale image applied, then its using it as is.

    Args:
        image (np.ndarray): The image in np format, either grayscale or RGB
        block_size (int, int): The size of the selected block window
        alpha (float): The alpha variance enhance parameter
        pi (float): The P factor enhanement
        delta (int): The threshold value to aply T1 or Gamma depending on the threshold
        bins (int): The number of bins

    Returns:
        (np.ndarray): The enhanced image using dual gamma
    """
    ndim = image.ndim
    R = bins - 1
    if R!= 255:
        raise ValueError(f"The range should be 256. This algorithm hasn't been tested on a different value, but given {bins}")
    if ndim == 3 and image.shape[2] == 3:
        hsv_image = rgb2hsv(image)
        gray_image = hsv_image[:, :, 2]
        gray_image = np.clip(gray_image * 255, 0, 255)
    elif ndim == 2:
        gray_image = image.copy()
    else:
        raise ValueError(f"Wrong number of shape or dimensions. Either single or 3 channel but image has shape {image.shape}")

    if isinstance(block_size, int):
        width_block = block_size
        height_block = block_size
        block_size = (block_size, block_size)
    elif isinstance(block_size, tuple):
        assert len(block_size) == 2, f"block_size dimension is not int or (tuple, tuple) but {len(block_size)}"
        height_block, width_block = block_size

    # Compute global histogram and global values
    glob_l_max = gray_image.max()
    glob_hist = np.histogram(gray_image, bins=bins)[0]
    glob_cdf = np.cumsum(glob_hist)
    glob_cdf = glob_cdf / glob_cdf[-1]
    glob_l_a = np.argwhere(glob_cdf > 0.75)[0]

    # Compute the padding values to pad the image
    pad_start_per_dim = [height_block // 2, width_block // 2]
    pad_end_per_dim = [(k - s % k) % k + int(np.ceil(k / 2.))
                       for k, s in zip(block_size, gray_image.shape)]

    # Pad the image with reflect mode for color invariance
    gray_image = np.pad(gray_image, [[p_i, p_f] for p_i, p_f in
                                     zip(pad_start_per_dim, pad_end_per_dim)],
                        mode='reflect')

    pad_height, pad_width = gray_image.shape[:2]
    n_height_blocks = int(pad_height / block_size[0])
    n_width_blocks = int(pad_width / block_size[1])
    hists = np.zeros((n_height_blocks, n_width_blocks, bins))
    beta_thresholds = np.zeros((n_height_blocks, n_width_blocks))
    result = np.zeros_like(gray_image)
    for ii in range(n_height_blocks):
        for jj in range(n_width_blocks):

            # Compute the block range and max block value
            max_val_block = gray_image[ii:ii + block_size[0], jj:jj + block_size[1]].max()
            r_block = max_val_block - gray_image[ii:ii + block_size[0], jj:jj + block_size[1]].min()

            hists[ii, jj] = np.histogram(gray_image[ii:ii + block_size[0], jj:jj + block_size[1]], bins=bins)[0]
            beta_thresholds[ii, jj] = compute_clip_limit(gray_image[ii:ii + block_size[0], jj:jj + block_size[1]],
                                                         alpha=alpha, pi=pi, R=R)
            hists[ii, jj] = clip_and_redistribute_hist(hists[ii, jj], beta_thresholds[ii, jj])

            pdf_min = hists[ii, jj].min()
            pdf_max = hists[ii, jj].max()

            weighted_hist = pdf_max * (hists[ii, jj] - pdf_min) / (pdf_max - pdf_min)
            weighted_cum_hist = np.cumsum(weighted_hist)
            pdf_sum = weighted_cum_hist[-1]
            weighted_cum_hist /= pdf_sum

            # Equalize histogram here!!!
            hists[ii, jj] = np.cumsum(hists[ii, jj])
            norm_cdf = hists[ii, jj] / hists[ii, jj, -1]

            # Compute Wen T1 and Gamma
            w_en = compute_w_en(l_max=glob_l_max, l_alpha=glob_l_a, cdf=norm_cdf)
            tau_1 = (max_val_block * w_en * norm_cdf)
            gamma = compute_gamma(l_max=glob_l_max, d_range=R, weighted_cdf=weighted_cum_hist)

            if r_block > delta:
                hists[ii, jj] = np.maximum(tau_1, gamma)
            else:
                hists[ii, jj] = gamma

    # Bilinear interpolation
    for i in range(pad_height):
        for j in range(pad_width):
            p_i = int(gray_image[i][j])

            # Get current block index
            block_x = i // block_size[0]
            block_y = j // block_size[1]

            # Get the central indexes of the running block
            center_block_x = block_x * block_size[0] + n_height_blocks // 2
            center_block_y = block_y * block_size[1] + n_width_blocks // 2

            # Compute the block coordinates to interpolate from
            block_y_a, block_y_c = compute_block_coords(center_block_x, block_x, i, n_height_blocks)
            block_x_a, block_x_b = compute_block_coords(center_block_y, block_y, j, n_width_blocks)

            # Block image coordinates
            y_a = block_y_c * block_size[0] + block_size[0] // 2
            y_c = block_y_a * block_size[0] + block_size[0] // 2
            x_a = block_x_a * block_size[1] + block_size[1] // 2
            x_b = block_x_b * block_size[1] + block_size[1] // 2

            m = compute_mn_factors((y_a, y_c), i)
            n = compute_mn_factors((x_a, x_b), j)

            Ta = hists[block_y_c, block_x_a, p_i]
            Tb = hists[block_y_c, block_x_b, p_i]
            Tc = hists[block_y_a, block_x_a, p_i]
            Td = hists[block_y_a, block_x_b, p_i]

            result[i, j] = int(m * (n * Ta + (1 - n) * Tb) + (1 - m) * (n * Tc + (1 - n) * Td))

    # Get the original shape of the image
    unpad_slices = tuple([slice(p_i, s - p_f) for p_i, p_f, s in
                          zip(pad_start_per_dim, pad_end_per_dim,
                              gray_image.shape)])
    result = result[unpad_slices]
    result = np.clip(result, 0, R)
    if ndim == 3:
        result = result / 255.
        hsv_image[:, :, 2] = result
        return (255 * hsv2rgb(hsv_image)).astype(np.uint8)
    else:
        return result.astype(np.uint8)


if __name__ == '__main__':
    path = './images/streets.jpg'
    image = io.imread(path)
    equalized_image = dual_gamma_clahe(image.copy(), block_size=32, alpha=100., delta=50., pi=1.5, bins=256)
    # equalized_image = dual_gamma_clahe(image.copy(), block_size=32, alpha=10., delta=70., pi=1.5, bins=256)
    # equalized_image = dual_gamma_clahe(image.copy(), block_size=64, alpha=80., delta=40., pi=3.5, bins=256)
    fig, ax = plt.subplots(1, 2)
    if image.ndim == 2:
        cmap = 'gray'
    else:
        cmap = None
    ax[0].imshow(image, cmap=cmap)
    ax[1].imshow(equalized_image, cmap=cmap)
    ax[0].set_title('Input Image')
    ax[1].set_title('Equalized Image ')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()
