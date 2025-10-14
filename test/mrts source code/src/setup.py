import os
import sys
import urllib.request
import tarfile
import zipfile
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')

def download_and_extract(url, extract_to):
    if not os.path.exists(extract_to):
        print(f"Downloading {url} ...")
        archive_path = os.path.join(src_dir, os.path.basename(url))
        os.makedirs(src_dir, exist_ok=True)
        urllib.request.urlretrieve(url, archive_path)
        print(f"Extracting {archive_path} ...")
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(src_dir)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(src_dir)
        else:
            print("Unknown archive format:", archive_path)
            sys.exit(1)
        os.remove(archive_path)

def find_or_install_eigen():
    # 你也可以改成檢查環境變數，這邊用簡化寫法
    eigen_path = os.path.join(src_dir, 'eigen-3.4.0')  # 修改版本號需同步修改
    if not os.path.isdir(eigen_path):
        download_and_extract('https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip', eigen_path)
    return eigen_path

def find_or_install_spectra():
    spectra_path = os.path.join(src_dir, 'spectra-1.1.0')  # 修改版本號需同步修改
    if not os.path.isdir(spectra_path):
        download_and_extract('https://github.com/yixuan/spectra/archive/refs/tags/v1.1.0.zip', spectra_path)
    return spectra_path

eigen_include = find_or_install_eigen()
spectra_include = os.path.join(find_or_install_spectra(), 'include')

setup(
    name='torch_decomp',
    ext_modules=[
        CppExtension(
            'torch_decomp',
            [os.path.join(root_dir, 'src', 'lanczos_algorithm.cpp')],
            include_dirs=[
                eigen_include,
                spectra_include,
            ],
            extra_compile_args=['-std=c++14'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
