% Quick check that the Octave port behaves: identical images must give
% MS-SSIM ~1 and LD ~0, and small shifts must give small LD.
%   octave --no-gui --quiet sanity.m <scan_dir>/
pkg load image
here = fileparts(mfilename('fullpath')); ev = fullfile(here, 'docunet_eval');
addpath(fullfile(here, 'compat')); addpath(ev); addpath(fullfile(ev, 'SIFTflow'));
addpath(fullfile(ev, 'SIFTflow', 'mexDenseSIFT')); addpath(fullfile(ev, 'SIFTflow', 'mexDiscreteFlow'));
args = argv(); scan = args{1};
ref = rgb2gray(imread(sprintf('%s1.png', scan)));
b = sqrt(598400/size(ref,1)/size(ref,2)); ref = imresize(ref, b);
tic; [ms, ld] = evalUnwarp(ref, ref); printf('identical:      MS-SSIM %.4f  LD %.3f  (%.1fs)\n', ms, ld, toc);
sh = circshift(ref, [0 5]); sh(:,1:5) = 255;
[ms, ld] = evalUnwarp(sh, ref); printf('shift 5 px:     MS-SSIM %.4f  LD %.3f\n', ms, ld);
other = imresize(rgb2gray(imread(sprintf('%s2.png', scan))), [size(ref,1), size(ref,2)]);
[ms, ld] = evalUnwarp(other, ref); printf('other document: MS-SSIM %.4f  LD %.3f\n', ms, ld);
disp('expected roughly: 1.0001/0.000, 0.8431/2.493, 0.3100/50.504');
