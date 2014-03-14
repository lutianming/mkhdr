import argparse
import os
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
    return images, times


def exposure_time(im):
    exif = im._getexif()
    a, b = exif[33434]      # EXIF exposuture time tag
    return float(a)/b


def recover_g(imgs, times):
    n_imgs = len(imgs)

    tmp = imgs[0].copy()
    tmp.thumbnail((20, 20))
    width, height = tmp.size
    n_pixels = width*height
    B = np.log(times)

    Z_R = np.zeros((n_pixels, n_imgs))
    Z_G = np.zeros((n_pixels, n_imgs))
    Z_B = np.zeros((n_pixels, n_imgs))
    # split rgb
    for i, im in enumerate(imgs):
        im = im.copy()
        im.thumbnail((width, height))
        r, g, b = im.split()
        Z_R[:, i] = np.array(r, dtype=np.int).flatten()
        Z_G[:, i] = np.array(g, dtype=np.int).flatten()
        Z_B[:, i] = np.array(b, dtype=np.int).flatten()

    l = 100
    g_r, lE_r = solve_g(Z_R, B, l, weight_function)
    g_g, lE_g = solve_g(Z_G, B, l, weight_function)

    g_b, lE_b = solve_g(Z_B, B, l, weight_function)
    x = np.arange(0, 256)
    plt.plot(g_r, x)
    plt.plot(g_g, x)
    plt.plot(g_b, x)
    plt.show()
    return g_r, g_g, g_b


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
    width, height = imgs[0].size
    length = width*height
    pixels = np.zeros((length, n_imgs), dtype=np.int8)
    tmp = np.zeros((length, n_imgs))
    weight = np.zeros((length, n_imgs))

    for i, im in enumerate(imgs):
        pixels[:, i] = np.array(im).flatten()

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
    return lnE.reshape((height, width))


def tone_mapping(E):
    p = E / (E+1) * 255
    return p


def save_hdr(filename, hdr):
    hdr.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help="""the directory that contains the
                        original images. If not specified,
                        the current working direcory is used.""")
    parser.add_argument('-o', '--output', default='hdr.jpg',
                        help='the output hdr image filename')
    parser.add_argument('-g', '--gui',
                        help='use gui interface', action='store_true')

    # parse arguments
    args = parser.parse_args()
    if args.directory:
        directory = args.directory
    else:
        directory = os.getcwd()
    output = args.output

    print('load images...')
    files = list_files(directory)
    images, times = read_images(files)

    print('recover g...')
    g_r, g_g, g_b = recover_g(images, times)
    R = []
    G = []
    B = []
    for img in images:
        r, g, b = img.split()
        R.append(r)
        G.append(g)
        B.append(b)

    w = gen_weight_map()
    print('radiance r')
    lnE_r = radiance(g_r, R, times, w)
    E_r = np.exp(lnE_r)
    p_r = tone_mapping(E_r)

    print('radiance g')
    lnE_g = radiance(g_g, G, times, w)
    E_g = np.exp(lnE_g)
    p_g = tone_mapping(E_g)

    print('radiance g')
    lnE_b = radiance(g_b, B, times, w)
    E_b = np.exp(lnE_b)
    p_b = tone_mapping(E_b)

    print('display result')
    r = Image.fromarray(np.array(p_r, dtype=np.uint8), mode='L')
    g = Image.fromarray(np.array(p_g, dtype=np.uint8), mode='L')
    b = Image.fromarray(np.array(p_b, dtype=np.uint8), mode='L')
    img = Image.merge('RGB', (r, g, b))
    img.show()

    img.save(output)
