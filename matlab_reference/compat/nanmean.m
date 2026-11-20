% Octave compatibility shim: core Octave has no nanmean (MATLAB does).
% Added to the path ONLY when running under Octave (see export_tuning_reference.m).
function m = nanmean(x, dim)
    if nargin < 2
        dim = find(size(x) ~= 1, 1);
        if isempty(dim), dim = 1; end
    end
    m = mean(x, dim, 'omitnan');
end
