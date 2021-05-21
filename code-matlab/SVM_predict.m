function [pred, accu] = SVM_predict(X, model, Y)
%SVM_PREDICT Use trained SVM model to predict the label of new points.
%   
%    [predictY, accu] = SVM_predict(X, model, Y)
%    
%    Input:
%        X: the testing data. Each row is a sample vector.
%        model: the trained SVM model generated from SVM_train.m
%        Y (optional): the label of testing data, used for compute accuracy.
%
%    Output:
%        pred: predict value of testing data.
%        accu: the accuracy of model in this test.
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.

testK = kernelDeal(X, model.sv, model.type, model.gamma, model.d);
pred = testK * model.svAlpha + model.b;

if nargin == 3
    predt = zeros(size(pred));
    predt(pred >= 0) = model.labelA;
    predt(pred <  0) = model.labelB;
    accu = sum(predt == Y) / length(Y) * 100;
end

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