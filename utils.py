import gdown
import os
import tarfile

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.webp', '.WEBP', '.TIF', '.tif', '.TIFF', '.tiff']

def extract(tar_path, target_path):
    try:
        tar = tarfile.open(tar_path, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, target_path)
        tar.close()
    except Exception  as e:
        print(e)

def from_gdrive_download(save_path = None):
    url = 'https://drive.google.com/uc?id=1WjWqCFOiPtEced1fIN3uzSNa3KlBLvep'
    output = 'hdc2021_0_9_last.h5.tar.gz'
    gdown.download(url, os.path.join(save_path, output),quiet=False)
    print('Download is okay!')

    url = 'https://drive.google.com/uc?id=1uofTwmzm42NH44ETRpaN9aolXh6sNR6r'
    output = 'hdc2021_10_19_last.h5.tar.gz'
    gdown.download(url, os.path.join(save_path, output),quiet=False)
    print('Download is okay!')

    # untar the download files
    extract(os.path.join(save_path, 'hdc2021_0_9_last.h5.tar.gz'),os.path.join(save_path))
    extract(os.path.join(save_path, 'hdc2021_10_19_last.h5.tar.gz'),os.path.join(save_path))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path, qualifier=is_image_file):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if qualifier(fname) and 'ref.jpg' not in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    if not images:
        print("Warning: {:s} has no valid image file".format(path))
    return images