from pathlib import Path

from setuptools import setup
import os
import platform
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
CSRC_DIR = os.path.join('fused_experts', 'C', 'csrc')

if os.environ.get('DEBUG'):
    extra_compile_args = ['-DDEBUG', '-O0']
else:
    extra_compile_args = ['-O3']

extra_compile_args.append('-std=c++17')

extra_link_args: list[str] = []

CUTLASS = ROOT / 'extern' / 'cutlass'

cutlass_includes = [
    str(CUTLASS / 'include'),
    str(CUTLASS / 'cute' / 'include'),
    str(CUTLASS / 'tools' / 'library' / 'include'),
]

# Add macOS specific flags
if platform.system() == 'Darwin':
    # Enable support for both Intel and Apple Silicon
    extra_compile_args.extend(['-arch', 'x86_64', '-arch', 'arm64'])
    extra_link_args.extend(['-arch', 'x86_64', '-arch', 'arm64'])

    # Add minimum deployment target for macOS
    extra_compile_args.append('-mmacosx-version-min=10.13')
    extra_link_args.append('-mmacosx-version-min=10.13')

extensions = [
    CUDAExtension(
        name='fused_experts.C.csrc.kernel',
        sources=[os.path.join(CSRC_DIR, 'kernel.cu')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=cutlass_includes,
    ),
]

setup(
    name='fused_experts',
    ext_modules=extensions,
    packages=['fused_experts', 'fused_experts.C.csrc'],
    package_data={
        'fused_experts.C.csrc': ['*.pyi'],  # Include .pyi files
    },
    cmdclass={'build_ext': BuildExtension},
)
