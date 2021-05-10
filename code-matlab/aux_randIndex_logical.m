function mask = aux_randIndex_logical(nSample, nSelected, t, filename)
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
%    >> mask = aux_randIndex_logical(10, 5, 3);
%    >> mask
%
%    mask =
% 
%      3x10 logical array
% 
%       0   0   0   1   0   1   0   1   1   1
%       0   0   1   1   0   1   1   0   0   1
%       0   0   1   1   0   1   0   1   1   0
%
%    Written by Junhong Zhang, SZU, with Matlab R2020a.    

% generate indeces
mask = false(t, nSample);
for ii = 1:t
    inds = randperm(nSample, nSelected);
    mask(ii, inds) = true;
end

% save the file
if nargin == 4
    clearvars -except inds rest filename;
    save(filename);
end

end