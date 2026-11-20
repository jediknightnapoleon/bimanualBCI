%% export_tuning_reference.m
% Reference oracle for the Python port (step 0.5 of the replication plan).
%
% Runs the COMPUTATION part of tuningAnalysis.m (lines 13-118, copied verbatim
% below; plotting is omitted) on one session and saves the tuning-curve
% matrices so `pixi run compare-reference` can check the Python code against them.
%
% Works in both MATLAB and GNU Octave. Run from the repo root:
%   octave-cli matlab_reference/export_tuning_reference.m
%   matlab -batch "run('matlab_reference/export_tuning_reference.m')"
%
% Input : Data/<dataName>.mat              (day 1750 = t5_06_02_2021 by default)
% Output: results/reference/tuning_ref_<engine>.mat containing
%         uniR_tc, uniL_tc, biR_tc, biL_tc (N channels x 8 directions),
%         conditions (trials x 3: [moveType, RH target ID, LH target ID]).

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(here);
addpath(fullfile(here, 'helpers'));

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave
    addpath(fullfile(here, 'compat'));  % nanmean shim
    engine = 'octave';
else
    engine = 'matlab';
end

dataPath = fullfile(repoRoot, 'Data');
dataName = 't5_06_02_2021';
fprintf('Loading %s ...\n', fullfile(dataPath, [dataName '.mat']));
dat = load(fullfile(dataPath, [dataName '.mat']));

%% ---- verbatim from tuningAnalysis.m lines 13-118 ----
% remove block-wise means
blkNums = [dat.blockNum];
uniqueBlkNums = unique(blkNums);
tx_meanRemoved = zeros(size(dat.tx)); % initialize a new matrix to store block-wise mean removed tx features
for b = 1 : length(uniqueBlkNums)
    blkInds = find(blkNums == uniqueBlkNums(b));
    curr_tx = dat.tx(:,blkInds);
    tx_meanRemoved(:,blkInds) = curr_tx - nanmean(curr_tx,2);
end

features = tx_meanRemoved*50; % multiply by 50 to get into seconds (20ms * 50 = 1s)
N = size(features,1); % number of channels

%% Cosine tuning curves
window = [15 35]; % window (in # of 20ms bins) relative to go cue to avg. spikes for tuning curve, e.g., [15 35]->[300 700]ms window after go cue  

% discretize the 2D target space into 8 equally sized wedges for trial-averaging
targetMapping = [157:201; 202:246; 247:291; 292:336; [337:359 0:21]; 22:66; 67:111; 112:156]'; % maps a target's location (position vector's angle wrt x-axis) to a wedge ID'd by column index 

% define the trial condition in the following form:
% [move type, right hand target ID, left hand target ID]
% move type : 1 = unimanual right trial, 2 = unimanual left trial, 3 = bimanual trial
% target ID is the corresponding column index of targetMapping
conditions = zeros(numel(dat.goCue),3); % initialize n-dimensional array
for trial = 1 : numel(dat.goCue)
    curr_goCue = dat.goCue(trial); % get current trial's go cue
    currTarg = dat.Tp(:,curr_goCue); % get position of both targets 
    currCursor = dat.Cp(:,curr_goCue); % get position of both cursors

    positionError = currTarg - currCursor; % compute the vector pointing from each cursor to its target
    TargID_rightHand = whichTarget_bimanual(targetMapping,positionError(1:2,:)'); % get the target ID for the right hand
    TargID_leftHand = whichTarget_bimanual(targetMapping,positionError(3:4,:)'); % get the target ID for the left hand

    mvType = dat.moveType(trial); % move type can be pulled from data structure
    if mvType == 1 % if a unimanual right hand trial, force left hand target ID to 9
        TargID_leftHand = 9;
    elseif mvType == 2 % do the same for the right hand if uni left trial
        TargID_rightHand = 9;
    end
    conditions(trial,:) = [mvType TargID_rightHand TargID_leftHand];
end

% tile features matrix so that we can easily access windows of activity for
% each desired factorization
featureVals = computeFeatureMatrices(features', conditions(:,2:3), dat.goCue, window);
% featureVals : N x RH x LH x T x Trials

% collect all time bins in each window for all 8 movement directions
% unimanual case only
uniR_tc = [];
uniL_tc = [];
for d = 1 : 8 % iterate over movement directions
    curr_uniR_featvals = squeeze(featureVals(:,d,9,:,:)); % extract data associated with unimanual right trials only (F2 of 9 means 'no movement' for left hand)
    curr_uniL_featvals = squeeze(featureVals(:,9,d,:,:)); % do the same for unimanual left trials (F1 of 9 means 'no movement' for right hand)

    % tile all bins
    tiled_uniR = [];
    tiled_uniL = [];
    for t = 1 : size(curr_uniR_featvals,3)
        tiled_uniR = [tiled_uniR squeeze(curr_uniR_featvals(:,:,t))];
        tiled_uniL = [tiled_uniL squeeze(curr_uniL_featvals(:,:,t))];
    end

    % compute the mean across all collected time bins 
    uniR_tc = [uniR_tc nanmean(tiled_uniR,2)];
    uniL_tc = [uniL_tc nanmean(tiled_uniL,2)];
end
% mean-subtract to center about zero
uniR_tc = uniR_tc - mean(uniR_tc,2); % uniR_tc = unimanual right hand tuning curve 
uniL_tc = uniL_tc - mean(uniL_tc,2); % uniL_tc = unimanual left hand tuning curve

% now for the bimanual context
inds_bi = find(conditions(:,1) == 3);
goCues_bi = dat.goCue(inds_bi); % get go cues for the bimanual trials only
conds_bi_R = conditions(inds_bi,2); % extract movement conditions for the right hand during bimanual context
featureVals_biR = computeFeatureMatrices(features', conds_bi_R, goCues_bi, window);

% extract movement conditions for the left hand during bimanual context
conds_bi_L = conditions(inds_bi,3);
featureVals_biL = computeFeatureMatrices(features', conds_bi_L, goCues_bi, window);

% collect all time bins in each window for all 8 movement directions
biR_tc = [];
biL_tc = [];
for d = 1 : 8
    curr_biR_featvals = squeeze(featureVals_biR(:,d,:,:));
    curr_biL_featvals = squeeze(featureVals_biL(:,d,:,:));

    % tile all 20ms bins
    tiled_biR = [];
    for t = 1 : size(curr_biR_featvals,3)
        tiled_biR = [tiled_biR squeeze(curr_biR_featvals(:,:,t))];
    end

    tiled_biL = [];
    for t = 1 : size(curr_biL_featvals,3)
        tiled_biL = [tiled_biL squeeze(curr_biL_featvals(:,:,t))];
    end

    % average across all collected time bins
    biR_tc = [biR_tc nanmean(tiled_biR,2)];
    biL_tc = [biL_tc nanmean(tiled_biL,2)];
end
% mean-subtract to center about zero
biR_tc = biR_tc - mean(biR_tc,2); % BiR_tc = bimanual right hand tuning curve
biL_tc = biL_tc - mean(biL_tc,2); % BiL_tc = bimanual left hand tuning curve
%% ---- end verbatim block ----

outDir = fullfile(repoRoot, 'results', 'reference');
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, ['tuning_ref_' engine '.mat']);
% -v7 so scipy.io.loadmat can read it
save(outFile, 'uniR_tc', 'uniL_tc', 'biR_tc', 'biL_tc', 'conditions', '-v7');
fprintf('Saved %s\n', outFile);
