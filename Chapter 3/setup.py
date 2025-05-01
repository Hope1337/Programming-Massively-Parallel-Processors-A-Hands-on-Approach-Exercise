from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vector_add',
    ext_modules=[
        CUDAExtension('hihi', [
            'lablib/vector_add.cpp',
            'lablib/vector_add_cuda.cu',
        ]),   
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

