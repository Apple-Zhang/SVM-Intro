function model = SVM_train(Y, X, param, kernel_type)
%SVM_TRAIN: simplified SVM implement.
%
%   model = SVM_train(Y, X, param, kernel_type)
%
%   Input:
%       Y: n*1 vector with binary value.
%       X: n*m matrix with n samples, where each sample has m dimensions.
%       kernel_type: use which kernel function, including 'linear', 'rbf',
%       'poly'.
%       param: parameters of kernel function, including c, gamma, d.
%       Each kernel is defined as:
%                      K_RBF(x, y)  = exp(-gamma * ||x-y||^2);
%                      K_Poly(x, y) = (x'*y + gamma) ^ d;
%   Output:
%       model: the trained SVM model.
%
%   Note: To run this, Matlab Optimization toolbox should be installed.
%   Written by Junhong Zhang, SZU, with Matlab R2020a.
%
EPSILON = 1e-13;
if nargin == 3
    kernel_type = 'linear';
end

if size(Y, 2) ~= 1
    error('Y should be a column vector with binary value.');
end

uY = unique(Y);
if length(uY) ~= 2
    error('only binary classification is supported.');
end

model.labelA = max(uY); % positive label
model.labelB = min(uY); % negative label
model.type = lower(kernel_type);

Y = sign(Y - sum(uY) / 2); % transform Y into {-1, 1}.
tempY = Y * Y';

% compute kernel matrix
switch model.type
    case 'linear'
        Q = tempY .* (X * X');
        model.d = nan;
        model.gamma = nan;
    case 'poly'
        Q = tempY .* (X * X' + param.gamma) .^ param.d;
        model.d = param.d;
        model.gamma = param.gamma;
    case 'rbf'
        Q = tempY .* exp( ...
            -param.gamma .* squareform(pdist(X, 'squaredeuclidean')) ...
        );
        model.d = nan;
        model.gamma = param.gamma;
    otherwise
        error('Invalid kernel type.');
end
clear tempY;

% compute alpha
tempOnes = ones(size(Y));
tempZeros = zeros(size(Y));

opt = optimoptions('quadprog', 'Display', 'off');
alpha = quadprog(Q, -tempOnes, [], [], Y', 0, tempZeros, param.c * tempOnes, [], opt);

model.svIndex = find(alpha > EPSILON);
model.svAlpha = alpha(model.svIndex) .* Y(model.svIndex);

clear temoOnes tempZeros alpha;

% compute bias
supportVectors = X(model.svIndex, :);
supportY = Y(model.svIndex);
b = mean(supportY - ...
    kernelDeal(supportVectors, supportVectors, model.type, model.gamma, model.d) * ...
    model.svAlpha ...
);
model.b = b;
model.sv = supportVectors;
clear supportVectors supportY;

end

function Z = kernelDeal(X1, X2, type, gamma, d)
switch type
    case 'linear'
        Z = (X1 * X2');
    case 'poly'
        Z = (X1 * X2' + gamma) .^ d;
    case 'rbf'
        Z = exp(...
            -gamma .* pdist2(X1, X2, 'squaredeuclidean') ...
        );
end

end
