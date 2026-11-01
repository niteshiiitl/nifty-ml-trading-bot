#!/usr/bin/env bash
set -e

echo "Downloading TA-Lib..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib

echo "Compiling TA-Lib locally..."
./configure --prefix=/opt/render/project/src/ta-lib-local
make
make install
cd ..

echo "Installing Python requirements..."
export TA_LIBRARY_PATH=/opt/render/project/src/ta-lib-local/lib
export TA_INCLUDE_PATH=/opt/render/project/src/ta-lib-local/include
pip install -r requirements.txt