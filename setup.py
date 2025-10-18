from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_extension",
    ext_modules=[
        CUDAExtension(
            name="my_extension",
            sources=["binding.cpp", "my_kernel.cu"],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=0"],  # 0: old ABI, 1: new ABI
                "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=0"],
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
