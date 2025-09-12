# CHPC dependencies
module load cmake cuda gcc zlib openmpi

cd AMGX
mkdir -p build
mkdir -p install
cd build

# configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH="80" \
  -DCMAKE_NO_MPI=OFF \
  -DCMAKE_INSTALL_PREFIX=../install

# build
make -j16 all
make install
