import gdown
import os
import tarfile

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
    url = 'https://drive.google.com/uc?id=1K6x7WjNfwmYscp8jFTV2hhn2ISkeHfJm'
    output = 'hdc2021_0_9_best.h5.tar.gz'
    gdown.download(url, os.path.join(save_path, output),quiet=False)
    print('Download is okay!')

    url = 'https://drive.google.com/uc?id=1yIc0KjZOzLG5rxftOH59aOCWErRQ06Au'
    output = 'hdc2021_10_19_best.h5.tar.gz'
    gdown.download(url, os.path.join(save_path, output),quiet=False)
    print('Download is okay!')

    # untar the download files
    extract(os.path.join(save_path, 'hdc2021_0_9_best.h5.tar.gz'),os.path.join(save_path))
    extract(os.path.join(save_path, 'hdc2021_10_19_best.h5.tar.gz'),os.path.join(save_path))