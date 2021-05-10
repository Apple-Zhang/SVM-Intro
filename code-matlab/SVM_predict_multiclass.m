function [pred, accu] = SVM_predict_multiclass(X, models, Y)
%SVM_PREDICT_MULTICLASS SVM predict model with multiclass.
%   
%    [predictY, accu] = SVM_predict_multiclass(X, models, Y)
%    
%    Input:
%        X: the testing data. Each row is a sample vector.
%        model: the trained SVM model generated from SVM_train_multiclass.m
%        Y (optional): the label of testing data, used for compute accuracy.
%
%    Output:
%        pred: predict value of testing data.
%        accu: the accuracy of model in this test.
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.

nModels = length(models);
nClasses = ceil(sqrt(nModels * 2));
nSamples = size(X, 1);

ctg = zeros(nSamples, nClasses);
% use 1v1 method.
for i = 1:nModels
    [prd, ~] = SVM_predict(X, models(i), Y);
    % vote for each sample
    for j = 1:nSamples
        ctg(j, prd(j)) = ctg(j, prd(j)) + 1;
    end
end

% winners are most voted.
[~, pred] = max(ctg, [], 2);

if nargin == 3
    accu = sum(Y == pred) / length(Y);
end

end

