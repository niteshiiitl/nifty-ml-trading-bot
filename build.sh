#!/usr/bin/env bash
# Render build script.
#
# Installs the native TA-Lib C library, then installs the Python dependencies.
# This is the main deployment risk: the `TA-Lib` Python wheel compiles against
# the C library, so the C library must be present before `pip install`.
#
# Render runs build scripts on a Linux (Ubuntu) image with apt available.
# Adapt the apt line if deploying to a different base image.

set -euo pipefail

echo "=== build.sh: installing build dependencies ==="
# build-essential: gcc/make to compile TA-Lib C lib and the Python wheel.
# wget: fetch the TA-Lib source tarball.
apt-get update -y
apt-get install -y build-essential wget

TA_LIB_VERSION="0.4.0"
TA_LIB_TARBALL="ta-lib-${TA_LIB_VERSION}-src.tar.gz"

echo "=== build.sh: building native TA-Lib C library (${TA_LIB_VERSION}) ==="
# Download, extract, configure, and install into /usr so the Python wheel
# build (pip install TA-Lib) can find the headers and shared object.
cd /tmp
wget -q "https://prdownloads.sourceforge.net/ta-lib/${TA_LIB_TARBALL}" -O "${TA_LIB_TARBALL}"
tar -xzf "${TA_LIB_TARBALL}"
cd "ta-lib"
# Prefix to /usr so no extra LD_LIBRARY_PATH is needed at runtime.
./configure --prefix=/usr
make -j"$(nproc)"
make install
# Ensure the linker sees the freshly built library on this build image.
ldconfig

echo "=== build.sh: installing Python dependencies ==="
cd "$(dirname "$0")"
pip install --upgrade pip
pip install -r requirements.txt

echo "=== build.sh: done ==="
# NOTE: TA_LIBRARY_PATH / TA_INCLUDE_PATH may be required by some TA-Lib wheel
# builds. With the C library installed under /usr (in the default search path)
# the wheel build should find it automatically. If the wheel build fails on a
# different base image, export TA_LIBRARY_PATH=/usr/lib and
# TA_INCLUDE_PATH=/usr/include before `pip install`.
