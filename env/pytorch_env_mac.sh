export USE_CUDA=0
export CMAKE_PREFIX_PATH=$(dirname $(which cmake))/../
export MACOSX_DEPLOYMENT_TARGET=10.14.6 CC=clang CXX=clang++ 
export ARCHFLAGS="-arch x86_64"
