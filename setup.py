import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    sources = [
        "./prime_moe/csrc/torch_interface.cpp",
        "./prime_moe/csrc/kernels/fused_moe_kernel.cu",
    ]

setuptools.setup(
    name="prime_moe",
    packages=setuptools.find_packages(include=["prime_moe.*"]),
    version="0.0.1",
    ext_modules=[CUDAExtension(name="prime_moe._C", sources=sources)],
    cmdclass={"build_ext": BuildExtension},
)
