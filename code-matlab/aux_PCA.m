function [P, sValue] = aux_PCA(X, dim)
%AUX_PCA Simple PCA (Principal Component Analysis) implement
%   
%    [P, sValue] = aux_PCA(X, dim)
%
%    Input:
%        X: data matrix. Each sample is a column.
%        dim: dimension of depressed sample.
%
%    Output:
%        P: projection matrix.
%        sValue (optional): each sigular value.
%
%    Written by Junhong Zhang, with Matlab R2020a.

cX = normalize(X, 'center', 'mean');
[U, S, ~] = svd(cX, 'econ');
P = U(:, 1:dim);

if nargout == 2
    sValue = diag(S);
    sValue = sValue(1:dim);
end


end

