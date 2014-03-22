import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def list_files(dir):
    full_path = [os.path.join(dir, f) for f in os.listdir(dir)]
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
    times = [exposure_time(im) for im in images]
    images = [np.array(im, dtype=np.int) for im in images]
    return images, times


def exposure_time(im):
    exif = im._getexif()
    a, b = exif[33434]      # EXIF exposuture time tag
    return float(a)/b


def recover_g(imgs, times, index_x, index_y):
    n_imgs = len(imgs)

    n_samples = 200
    x, y = imgs[0].shape

    B = np.log(times)
    Z = np.zeros((n_samples, n_imgs))

    for i, im in enumerate(imgs):
        Z[:, i] = np.array(im[index_x, index_y], dtype=np.int).flatten()

    l = 1000
    g, lE = solve_g(Z, B, l, weight_function)
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
    w = [weight_function(z) for z in range(256)]
    return w


def radiance(g, imgs, times, w):
    n_imgs = len(imgs)
    width, height = imgs[0].shape
    length = width*height
    pixels = np.zeros((length, n_imgs), dtype=np.int8)

    for i, im in enumerate(imgs):
        pixels[:, i] = im.flatten()

    # for z in range(256):
    #     rows, cols = np.where(pixels == z)
    #     tmp[rows, cols] = g[z]
    #     weight[rows, cols] = w(z)
    # tmp = tmp - np.log(times)

    vfun = np.vectorize(lambda z: g[z])
    tmp = vfun(pixels)
    tmp = tmp - np.log(times)

    vfun = np.vectorize(lambda z: w[z])
    weight = vfun(pixels)

    lnE = np.sum(weight*tmp, axis=1) / np.sum(weight, axis=1)
    return lnE.reshape((width, height))


def tone_mapping(E):
    p = E / (E+1) * 255
    return p


def make_hdr(images, times):
    ndim = images[0].ndim
    w = gen_weight_map()
    x = images[0].shape[0]
    y = images[1].shape[1]
    if ndim == 3:
        colors = images[0].shape[2]
    else:
        colors = 1

    subs = []
    n_samples = 200
    index_x = np.random.randint(0, x, n_samples)
    index_y = np.random.randint(0, y, n_samples)
    gs = []
    for i in range(colors):
        subimgs = [im[:, :, i] for im in images]
        g = recover_g(subimgs, times, index_x, index_y)
        gs.append(g)
        lnE = radiance(g, subimgs, times, w)
        E = np.exp(lnE)
        p = tone_mapping(E)
        subimg = Image.fromarray(np.array(p, dtype=np.int8), mode='L')
        subs.append(subimg)

    x = np.arange(0, 256)
    for g in gs:
        plt.plot(g, x)
    plt.show()
    img = Image.merge('RGB', (subs[0], subs[1], subs[2]))
    return img
