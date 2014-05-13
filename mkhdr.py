import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filter import denoise_bilateral

LAMBDA = 'lambda'
SMOOTH = 'smooth'
TONE_MAPPING_OP = 'tone_mapping_op'
SIGMA_R = 'sigma_r'
SIGMA_D = 'sigma_d'
A = 'a'
SATURATION = 'saturation'


def list_files(d):
    full_path = [os.path.join(d, f) for f in os.listdir(d)]
    files = [f for f in full_path
             if os.path.isfile(f)]
    return files


def read_images(filenames):
    names = {os.path.basename(f): f for f in filenames}

    if "config.txt" in names:
        timedic = {}
        #read times
        config = names["config.txt"]
        with open(config, "r") as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(" ")
                timedic[tokens[0]] = float(tokens[1])
        imagedic = {}
        for k, v in names.items():
            if k == "config.txt":
                continue
            image = Image.open(v)
            imagedic[k] = image
        images = []
        times = []
        for k in imagedic.keys():
            images.append(np.array(imagedic[k], dtype=np.int))
            times.append(timedic[k])
    else:
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

    n_samples = index_x.size
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
    return 0.27*R+0.67*G+0.06*B


def local_durand(E, sigma_r=0.4, sigma_d=50):
    intensity = 20*E[:, :, 0] + 40*E[:, :, 1] + E[:, :, 2]
    intensity = intensity/61
    # intensity = luminance(E[:, :, 0],
    #                       E[:, :, 1],
    #                       E[:, :, 2])
    x, y = intensity.shape
    n_channels = 3
    log_in_intensity = np.log10(intensity).astype(np.float32)
    log_base = cv2.bilateralFilter(log_in_intensity, 3, sigma_r, sigma_d)
    log_detail = log_in_intensity - log_base

    compressFactor = np.log10(50) / (np.max(log_base) - np.min(log_base))
    scaleFactor = np.max(log_base)*compressFactor

    log_out_intensity = log_base*compressFactor - scaleFactor + log_detail
    out_intensity = 10**(log_out_intensity)

    img = np.zeros((x, y, n_channels))
    for i in range(n_channels):
        # channel = np.power(E[:, :, i] / intensity, 0.6) * out_intensity
        # channel = channel * 255
        # channel[channel > 255] = 255
        # img[:, :, i] = channel

        channel = E[:, :, i] / intensity * out_intensity
        channel = channel/np.max(channel)
        img[:, :, i] = channel**(1/2.2) * 255
        # img[:, :, i] = (channel / (1 + channel))**(1/2.2) * 255
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
        channel = np.power(E[:, :, i] / L, saturation) * L_d
        channel = channel * 255
        channel[channel > 255] = 255
        img[:, :, i] = channel
    return img

def gamma_correct(img, gamma):
    img = np.power(img, gamma)
    return img

#tone mapping operator warppers so that they all except same args
tone_mapping_operators = {
    "global_simple":
    lambda E, args: global_simple(E),
    "global_reinhards":
    lambda E, args: global_reinhards(E, args['a'], args['saturation']),
    "local_durand":
    lambda E, args: local_durand(E, args['sigma_r'], args['sigma_d'])}


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
    y = images[0].shape[1]
    if ndim == 3:
        n_channels = images[0].shape[2]
    else:
        n_channels = 1

    n_samples = args['samples']
    nsqrt = int(np.sqrt(n_samples))
    border = 10
    index_x = np.linspace(border, x-border, nsqrt).astype(np.int)
    index_y = np.linspace(border, y-border, nsqrt).astype(np.int)
    index_x, index_y = np.meshgrid(index_x, index_y)
    g = np.zeros((n_channels, 256))
    E = np.zeros((x, y, n_channels))

    for i in range(n_channels):
        subimgs = [im[:, :, i] for im in images]
        print("recover g")
        g[i, :] = recover_g(subimgs, times, index_x, index_y, args["lambda"])
        print("radiance map")
        E[:, :, i] = radiance_map(g[i, :], subimgs, times, w)

    print("tone mapping")
    tone_mapping = tone_mapping_operators[args["tone_mapping_op"]]

    img = tone_mapping(E, args)
    img = Image.fromarray(img.astype(np.int8), mode='RGB')
    # x = np.arange(0, 256)
    # for g in gs:
    #     plt.plot(g, x)
    # plt.show()
    return img, g
