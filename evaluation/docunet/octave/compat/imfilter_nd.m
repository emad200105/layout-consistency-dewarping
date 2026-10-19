function B = imfilter_nd(A, h, varargin)
% Octave's imfilter rejects H x W x 128 arrays (dense SIFT); MATLAB filters
% every channel independently with the 2-D kernel. Do exactly that.
  if ndims(A) <= 2 || size(A,3) <= 4
    B = imfilter(A, h, varargin{:});
    return
  end
  B = zeros(size(A), class(A));
  for c = 1:size(A,3)
    B(:,:,c) = imfilter(A(:,:,c), h, varargin{:});
  end
end
