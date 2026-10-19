% MS-SSIM and Local Distortion (LD) on the DocUNet benchmark, in GNU Octave.
%
% Usage (from evaluation/docunet/octave, after setup_octave_eval.sh):
%   octave --no-gui --quiet eval_dir.m <rec_dir>/ <tail> <out_csv> <scan_dir>/ [n_docs]
%
%   Reads <rec_dir>/<i>_<k><tail> for documents i = 1..65 (or 1..n_docs) and
%   photos k = 1,2, and the flatbed scan <scan_dir>/<i>.png.
%
% Follows DocTr's ssim_ld_eval.m exactly:
%   grayscale; the scan is resized to an area of 598,400 pixels; the rectified
%   image is resized to the scan's size; then evalUnwarp (from the official
%   DocUNet eval.zip) gives MS-SSIM and LD.
% Resumable: images already in <out_csv> are skipped.
pkg load image
here = fileparts(mfilename('fullpath'));
ev = fullfile(here, 'docunet_eval');           % created by setup_octave_eval.sh
addpath(fullfile(here, 'compat'));
addpath(ev); addpath(fullfile(ev, 'SIFTflow'));
addpath(fullfile(ev, 'SIFTflow', 'mexDenseSIFT')); addpath(fullfile(ev, 'SIFTflow', 'mexDiscreteFlow'));
args = argv();
rec_dir = args{1}; tail = args{2}; out_csv = args{3}; path_scan = args{4};
n_docs = 65; if numel(args) >= 5, n_docs = str2double(args{5}); end
tarea = 598400;
done = {};
if exist(out_csv, 'file')
  fid = fopen(out_csv); fgetl(fid);
  while true
    l = fgetl(fid); if ~ischar(l), break; end
    p = strsplit(l, ','); done{end+1} = p{1};
  end
  fclose(fid);
  fid = fopen(out_csv, 'a');
else
  fid = fopen(out_csv, 'w'); fprintf(fid, 'name,ms_ssim,ld\n');
end
for i = 1:n_docs
  ref = imread(sprintf('%s%d.png', path_scan, i));
  ref = rgb2gray(ref);
  b = sqrt(tarea/size(ref,1)/size(ref,2));
  ref = imresize(ref, b);
  for k = 1:2
    name = sprintf('%d_%d', i, k);
    if any(strcmp(done, name)), continue; end
    A = imread(sprintf('%s%d_%d%s', rec_dir, i, k, tail));
    if size(A,3) == 4, A = A(:,:,1:3); end
    if size(A,3) == 3, A = rgb2gray(A); end
    A = imresize(A, [size(ref,1), size(ref,2)]);
    [ms, ld] = evalUnwarp(A, ref);
    fprintf(fid, '%s,%.6f,%.6f\n', name, ms, ld); fflush(fid);
    printf('%s  MS-SSIM %.4f  LD %.3f\n', name, ms, ld); fflush(stdout);
  end
end
fclose(fid);
disp('EVAL_DONE');
