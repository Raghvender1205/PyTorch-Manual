from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='convolution', 
        ext_modules=[cpp_extension.CppExtension('convolution', ['convolution.cpp'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})   

Extension(
    name='convolution',
    sources=['convolution.cpp'], 
    include_dirs=cpp_extension.include_paths(), language='c++')