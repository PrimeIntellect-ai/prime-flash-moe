import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ.get(
        'TORCH_CUDA_ARCH_LIST',
        '10.0a',
    )

    sources = [
        './prime_moe/csrc/torch_interface.cpp',
        './prime_moe/csrc/kernels/fused_moe_bf16_kernel.cu',
        './prime_moe/csrc/kernels/fused_moe_mxfp8_kernel.cu',
    ]

CPP_STD: int = 20 # Use c++ 20 :)
assert CPP_STD in (14, 17, 20, 26)

setuptools.setup(
    name='prime_moe',
    packages=setuptools.find_packages(include=['prime_moe', 'prime_moe.*']),
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name='prime_moe._C',
            sources=sources,
            extra_compile_args={
                'cxx': [f'-std=c++{CPP_STD}'],
                'nvcc': [f'-std=c++{CPP_STD}'],
            },
        )],
    cmdclass={'build_ext': BuildExtension},
)
