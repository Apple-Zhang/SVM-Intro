function [W, eigv] = aux_LDA(X, Y, dim)
%AUX_LDA Simple LDA (Linear Discriminant Analysis) implement
%   
%    [W, eigv] = aux_LDA(X, Y, dim)
%
%    Input:
%        X: training data matrix. Each sample is a column.
%        Y: label vector.
%        dim: dimension of depressed sample.
%
%    Output:
%        W: projection matrix.
%        eigv (optional): eigenvalues of the eigen-decompression in LDA.
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.

uY = unique(Y)';
nClass = length(uY);
[nDim, nSample] = size(X);
if nClass < 2
    error("Invalid input Y. The number of class should greater than 2.");
end

% If St is singular, then use PCA to reduce dimension.
PCAflag = false;
if nDim > nSample
    % ues PCA to reduce dimension.
    P = aux_PCA(X, nSample-1);
    X = P' * X;
    
    PCAflag = true;
    nDim = nSample-1;
end

% Compute St
mu = mean(X, 2);
St = X*X' - mu*mu';

% Compute Sb
nSampleInClass = zeros(1, nClass);
nMeanInClass = zeros(nDim, nClass);
for ii = uY
    mask = (Y == ii);
    nSampleInClass(ii) = sum(mask);
    nMeanInClass(:, ii) = mean(X(:, mask), 2);
end
Sb = nSampleInClass .* nMeanInClass*nMeanInClass' - nSample * mu*mu';

% LDA projection matrix
if dim < 0 || dim >= nClass
    dim = nClass-1;
end

[V, D] = eig(Sb, St);
[dsort, idx] = sort(diag(D), 'descend');
U = V(:, idx(1:dim));
U = real(U);

if nargout == 2
    eigv = dsort(1:dim);
end

% Check if PCA is used.
if PCAflag
    W = P * U;
else
    W = U;
end

end