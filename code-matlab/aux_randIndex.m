function [inds, rest] = aux_randIndex(nSample, nSelected, t, filename)
%AUX_RANDINDEX: generate random index for training dataset.
%
%    inds = aux_randIndex(nSample, nSelected, times, filename)
%
%    Input:
%        nSample: the number of sample
%        nSelected: the number of selected as training sample
%        t: number of generation.
%        filename (optional): if you want to save the generated indeces,
%            use this parameter to specify the file name.
%
%    Output:
%        inds: the generated indeces, with each row as one time generation.
%        rest: rest part of the indeces.
%
%    Example: 
%    Suppose we have 10 samples. And only 5 samples are used as training
%    sets. We want to take the results of 3 times experiments. Then we can 
%    generate the index of TRAINING samples and TESTING samples as trainID
%    and testID, respectively:
%
%    >> [trainID, testID] = aux_randIndex(10, 5, 3);
%    >> disp(idx); disp(rest);
%    trainID =
%          2     7     6    10     3
%          5     2     7     4     1
%          6    10     9     7     2
%    testID =
%          4     8     1     9     5
%         10     8     3     9     6
%          8     5     3     4     1
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.    

% generate indeces
inds = zeros(t, nSelected);
if nargout == 1
    for ii = 1:t
        inds(ii, :) = int32(randperm(nSample, nSelected));
    end
else
    rest = zeros(t, nSample-nSelected);
    for ii = 1:t
        temp = int32(randperm(nSample));
        inds(ii, :) = temp(1:nSelected);
        rest(ii, :) = temp(nSelected+1:end);
    end
end

% save the file
if nargin == 4
    clearvars -except inds rest filename;
    save(filename);
end

end