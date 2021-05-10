function models = SVM_train_multiclass(Y, X, param, kernel_type)
%SVM_TRAIN_MULTICLASS: simplified multiclass SVM implement.
%   
%   models = SVM_train(Y, X, param, kernel_type)
%   
%   Input:
%       Y: n*1 vector with different values.
%       X: n*m matrix with n samples, where each sample has m dimensions.
%       kernel_type: use which kernel function, including 'linear', 'rbf',
%       'poly'.
%       param: parameters of kernel function, including c, gamma, d.
%       Each kernel is defined as:
%                      K_RBF(x, y)  = exp(-gamma * ||x-y||^2);
%                      K_Poly(x, y) = (x'*y + gamma) ^ d;
%   Output:
%       model: the trained SVM models.

ctg = unique(Y);
nClass = length(ctg);

models(nClass * (nClass-1)/2) = struct(...
    'labelA', [], ...
    'labelB', [], ...
    'type', [], ...
    'd', [], 'gamma', [], ...
    'svIndex', [], ...
    'sv', [], ...
    'svAlpha', [], ...
    'b', 0);
counter = 1;
for ii = 1:nClass
    for jj = ii+1:nClass
        Yij = Y(Y == ctg(ii) | Y == ctg(jj));
        Xij = X(Y == ctg(ii) | Y == ctg(jj), :);
        models(counter) = SVM_train(Yij, Xij, param, kernel_type);
        counter = counter + 1;
    end
end

end

