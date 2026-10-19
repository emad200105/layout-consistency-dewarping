#!/usr/bin/env bash
# Prepare the official DocUNet evaluation code (MATLAB) to run in GNU Octave.
#
#   bash setup_octave_eval.sh /path/to/eval.zip
#
# eval.zip is the official evaluation package from the DocUNet benchmark page
# (http://vision.cs.stonybrook.edu/~kema/docwarp/, together with crop.zip
# and scan.zip). It is not redistributed
# here; this script unpacks it, applies our Octave patch and compiles the
# SIFT-flow MEX files.
#
# Needs: octave (tested 8.4), its image package (tested 2.14), mkoctfile
# (Ubuntu: apt install octave octave-image octave-dev), patch, unzip, g++.
set -euo pipefail
ZIP=${1:?usage: setup_octave_eval.sh /path/to/eval.zip}
HERE=$(cd "$(dirname "$0")" && pwd)
DST="$HERE/octave/docunet_eval"
rm -rf "$DST"; mkdir -p "$DST"
unzip -q "$ZIP" -d "$DST"
cd "$DST"
# The patch:
#  - project.h: compile for Linux/macOS, not with OpenCV/Visual Studio
#  - int -> mwSize for mxGetDimensions / mxCreateNumericArray (64-bit API)
#  - #include "vector.h" -> "Vector.h" (case-sensitive file systems)
#  - SIFTflowc2f.m: imfilter -> imfilter_nd, because Octave's imfilter refuses
#    the 128-channel dense-SIFT array that MATLAB filters channel by channel
patch -p1 < "$HERE/octave/siftflow_octave.patch"
(cd SIFTflow/mexDenseSIFT && mkoctfile --mex -O2 mexDenseSIFT.cpp Matrix.cpp Vector.cpp)
(cd SIFTflow/mexDiscreteFlow && mkoctfile --mex -O2 mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp)
ls -l SIFTflow/mexDenseSIFT/*.mex SIFTflow/mexDiscreteFlow/*.mex
echo "OK - now run:  cd $HERE/octave && octave --no-gui --quiet sanity.m <scan_dir>/"
