# CHPC dependencies
module load cmake cuda gcc zlib openmpi

# clone amgx(Branch off at V2.4.0)
git clone --recursive git@github.com:nvidia/amgx.git
cd amgx
git checkout v2.4.0
mkdir -p build
mkdir -p install
cd build
# get current directory
amgx_source_dir=$(pwd)
echo "amgx_source_dir: $amgx_source_dir"

# configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH="80" \  # A100, A30
  -DCMAKE_NO_MPI=OFF \
  -DCMAKE_INSTALL_PREFIX=$(amgx_source_dir)/install

# build
make -j16 all
make install
