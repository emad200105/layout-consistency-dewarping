function [ssimval, ssimmap] = ssim(A, ref)
% MATLAB R2019a-compatible SSIM with default settings, for Octave.
%   Gaussian window, std 1.5, size 2*ceil(3*1.5)+1 = 11, 'conv' + 'replicate'.
%   C1 = (0.01 L)^2, C2 = (0.03 L)^2, C3 = C2/2, exponents [1 1 1],
%   L = dynamic range of the input class (255 for uint8), computed in double.
%   Returns mean of the SSIM map, as MATLAB does.
  if isinteger(A)
    L = double(intmax(class(A))) - double(intmin(class(A)));
  else
    L = 1;
  end
  A = double(A); ref = double(ref);
  radius = 1.5;
  filtRadius = ceil(radius*3);
  filtSize = 2*filtRadius + 1;
  g = fspecial('gaussian', [filtSize filtSize], radius);
  C1 = (0.01*L)^2; C2 = (0.03*L)^2;
  mux = imfilter(A, g, 'conv', 'replicate');
  muy = imfilter(ref, g, 'conv', 'replicate');
  muxy = mux.*muy; mux2 = mux.^2; muy2 = muy.^2;
  sigmax2 = imfilter(A.^2, g, 'conv', 'replicate') - mux2;
  sigmay2 = imfilter(ref.^2, g, 'conv', 'replicate') - muy2;
  sigmaxy = imfilter(A.*ref, g, 'conv', 'replicate') - muxy;
  num = (2*muxy + C1).*(2*sigmaxy + C2);
  den = (mux2 + muy2 + C1).*(sigmax2 + sigmay2 + C2);
  ssimmap = num./den;
  ssimval = mean(ssimmap(:));
end
