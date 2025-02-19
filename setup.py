from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_kernels',
    ext_modules=[
        CUDAExtension(
            'custom_kernels', 
            ['simplenn_cuda.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)


