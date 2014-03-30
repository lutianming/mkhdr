import os
import sys
from PIL import Image
from skimage import filter
import cv2
import numpy as np
import matplotlib.pyplot as plt


def list_files(d):
    full_path = [os.path.join(d, f) for f in os.listdir(d)]
    files = [f for f in full_path
             if os.path.isfile(f)]
    return files


def read_images(filenames):
    images = []
    for f in filenames:
        image = Image.open(f)
        images.append(image)

    # sort by image exposure time
    images.sort(key=exposure_time)
    times = np.array([exposure_time(im) for im in images])
    images = [np.array(im, dtype=np.int) for im in images]
    return images, times


def exposure_time(im):
    exif = im._getexif()
    a, b = exif[33434]      # EXIF exposuture time tag
    return float(a)/b


def recover_g(imgs, times, index_x, index_y, smooth_factor=50):
    n_imgs = len(imgs)

    n_samples = 200
    x, y = imgs[0].shape

    B = np.log(times)
    Z = np.zeros((n_samples, n_imgs))

    for i, im in enumerate(imgs):
        Z[:, i] = np.array(im[index_x, index_y], dtype=np.int).flatten()

    g, lE = solve_g(Z, B, smooth_factor, weight_function)
    return g


def solve_g(Z, B, l, w):
    n = 256

    A = np.zeros((np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 0)))
    b = np.zeros(np.size(A, 0))

    k = 0
    for i in range(np.size(Z, 0)):
        for j in range(np.size(Z, 1)):
            wij = w(Z[i, j])
            A[k, Z[i, j]] = wij
            A[k, n+i] = -wij
            b[k] = wij * B[j]
            k = k+1

    A[k, 128] = 1
    k = k+1

    for i in range(n-2):
        A[k, i] = l*w(i+1)
        A[k, i+1] = -2*l*w(i+1)
        A[k, i+2] = l*w(i+1)
        k = k+1

    x = np.linalg.lstsq(A, b)[0]
    g = x[0:n]
    lE = x[n:np.size(x, 0)]
    return g, lE


def weight_function(z):
    z_min = 0
    z_max = 255
    z_mean = 128

    if z <= z_mean:
        return z - z_min + 1
    else:
        return z_max - z + 1


def gen_weight_map():
    w = np.array([weight_function(z) for z in range(256)])
    return w


def radiance_map(g, imgs, times, w):
    n_imgs = len(imgs)
    width, height = imgs[0].shape
    length = width*height
    pixels = np.zeros((length, n_imgs), dtype=np.int8)

    for i, im in enumerate(imgs):
        pixels[:, i] = im.flatten()

    tmp = g[pixels] - np.log(times)
    weight = w[pixels]
    weight[pixels == 255] = 0

    lnE = np.sum(weight*tmp, axis=1) / np.sum(weight, axis=1)
    return np.exp(lnE.reshape((width, height)))


def global_simple(E):
    p = E / (E+1) * 255
    return p


def luminance(R, G, B):
    return 0.2989*R+0.5866*G+0.1145*B


def local_durand(E, sigma_r=0.4, sigma_d=100):
    intensity = luminance(E[:, :, 0],
                          E[:, :, 1],
                          E[:, :, 2])

    x, y = intensity.shape
    n_channels = 3
    log_in_intensity = np.log(intensity).astype(np.float32)
    # log_base = filter.denoise_bilateral(log_in_intensity, sigma_r, sigma_d)
    log_base = cv2.bilateralFilter(log_in_intensity, 5, 50, 50)
    log_detail = log_in_intensity - log_base

    compressFactor = np.log(50) / (np.max(log_base) - np.min(log_base))
    scaleFactor = np.max(log_base)*compressFactor

    log_out_intensity = log_base*compressFactor - scaleFactor + log_detail

    img = np.zeros((x, y, n_channels))
    for i in range(n_channels):
        channel = E[:, :, i] / intensity
        channel = channel*np.exp(log_out_intensity)
        img[:, :, i] = channel / (1 + channel) * 255
    return img


def global_reinhards(E, a=0.48, saturation=0.6):
    delta = 0.0001

    L = luminance(E[:, :, 0],
                  E[:, :, 1],
                  E[:, :, 2])

    x = L.shape[0]
    y = L.shape[1]
    L_w = np.exp(np.sum(np.log(delta + L)) / (x*y))
    L_xy = a * (L/L_w)
    L_d = L_xy / (1 + L_xy)

    img = np.zeros((x, y, 3))
    for i in range(3):
        channel = np.power(E[:, :, i] / L, saturation) * L_d * 255
        channel[channel > 255] = 255
        img[:, :, i] = channel
    return img

tone_mapping_operators = {
    "global_simple": global_simple,
    "global_reinhards": global_reinhards,
    "local_durand": local_durand,
}


def default_args(args):
    if not args:
        args = {}
    if "lambda" not in args:
        args["lambda"] = 50
    if "tone_mapping_op" not in args:
        args["tone_mapping_op"] = "global_reinhards"
    return args


def make_hdr(images, times, args=None):
    args = default_args(args)
    ndim = images[0].ndim
    w = gen_weight_map()
    x = images[0].shape[0]
    y = images[1].shape[1]
    if ndim == 3:
        n_channels = images[0].shape[2]
    else:
        n_channels = 1

    n_samples = 200
    index_x = np.random.randint(0, x, n_samples)
    index_y = np.random.randint(0, y, n_samples)
    g = np.zeros((n_channels, 256))
    E = np.zeros((x, y, n_channels))

    for i in range(n_channels):
        subimgs = [im[:, :, i] for im in images]
        print("recover g")
        g[i, :] = recover_g(subimgs, times, index_x, index_y, args["lambda"])
        print("radiance_map")
        E[:, :, i] = radiance_map(g[i, :], subimgs, times, w)

    print("tone mapping")
    tone_mapping = tone_mapping_operators[args["tone_mapping_op"]]

    img = tone_mapping(E)
    img = Image.fromarray(img.astype(np.int8), mode='RGB')
    # x = np.arange(0, 256)
    # for g in gs:
    #     plt.plot(g, x)
    # plt.show()
    return img
