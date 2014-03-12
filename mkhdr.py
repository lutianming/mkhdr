import argparse
import os


def list_files(dir):
    files = [f for f in os.listdir(dir)
             if os.path.isfile(os.path.join(dir, f))]
    return files


def load_images(filenames):
    for f in filenames:
        print(f)


def write_hdr(filename, hdr):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help="""the directory that contains the
                        original images. If not specified,
                        the current working direcory is used.""")
    parser.add_argument('-o', '--output', default='hdr.jpg',
                        help='the output hdr image filename')

    args = parser.parse_args()

    if args.directory:
        directory = args.directory
    else:
        directory = os.getcwd()

        output = args.output

    print list_files(directory)
