#!/usr/bin/env bash
# Build Tesseract 5.0.1 from source into $PREFIX (default /opt/tess501).
#
# Why: OCR edit distance / CER on DocUNet is very sensitive to the Tesseract
# version. Published numbers (DocGeoNet, DocScanner, DocTr++) used 5.0.1;
# with it, our score for the unrectified photos matches theirs
# (CER 0.5092 vs 0.5089). Ubuntu 24.04's 5.3.4 gives 0.4835.
#
# The English model is the standard one from Ubuntu's tesseract-ocr-eng
# package (4.1.0), found via TESSDATA_PREFIX in the wrapper written below.
set -euo pipefail
PREFIX=${PREFIX:-/opt/tess501}
TESSDATA=${TESSDATA:-/usr/share/tesseract-ocr/5/tessdata}
sudo apt-get install -y git autoconf automake libtool pkg-config g++ make \
     libleptonica-dev tesseract-ocr-eng
WORK=$(mktemp -d)
git clone -q --depth 1 -b 5.0.1 https://github.com/tesseract-ocr/tesseract.git "$WORK/tesseract"
cd "$WORK/tesseract"
./autogen.sh
./configure --prefix="$PREFIX" --disable-openmp
make -j"$(nproc)"
sudo make install
sudo tee "$PREFIX/tess.sh" > /dev/null <<EOF
#!/bin/sh
export LD_LIBRARY_PATH=$PREFIX/lib
export TESSDATA_PREFIX=$TESSDATA
exec $PREFIX/bin/tesseract "\$@"
EOF
sudo chmod +x "$PREFIX/tess.sh"
"$PREFIX/tess.sh" --version | head -1
echo "Use it with:  --tesseract $PREFIX/tess.sh"
