function [pred, accu] = aux_KNNPredict(Xtest, Xtrain, Ytrain, k, Ytest)
%AUX_KNNPREDICT Simple implement of KNN classifer (K-Nearest Neighbour)
%   
%    [pred, accu] = aux_KNNPredict(Xtest, Xtrain, Ytrain, k, Ytest)
%
%    Input:
%        Xtest: the testing data matrix. Each column is a sample vector.
%        Xtrain: the training data matrix. Each column is a sample vector.
%        k: parameter of KNN classifier
%        Ytest (optional): the actual label of testing data.
%
%    Output:
%        pred: prediction labels of testing data.
%        accu (optional): recognition rate (accuracy).
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.

idx = knnsearch(Xtrain', Xtest', 'k', k);
chose = Ytrain(idx);
pred = mode(chose, 2);

if nargout == 2
    if nargin == 5
        accu = sum(pred == Ytest) / length(pred);
    else
        error("Too little parameters. If you want to output accu, Ytest is needed.");
    end
end

end

