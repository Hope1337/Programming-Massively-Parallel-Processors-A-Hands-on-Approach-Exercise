from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='Ex3',
    ext_modules=[
        CUDAExtension('Ex3', [
            'lablib/3a.cpp',
            'lablib/3a_cuda.cu',
        ]),   
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

