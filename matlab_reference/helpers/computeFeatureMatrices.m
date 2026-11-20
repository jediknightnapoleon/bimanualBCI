% Copied verbatim from tuningAnalysis.m (upstream d-r-deo/bimanualBCI) so the
% function can be called from a script under both MATLAB and Octave.

function featureVals = computeFeatureMatrices(features, trialCodes, eventIdx, timeWindow)
    % This function tiles the features matrix into a N x F1 x F2 x W x R tensor
    % N : # of channels
    % F1 : # of conditions for Factor 1 (in our case 8 movement directions with an additional 'No movement' condition for the right hand)
    % F2 : # of conditions for Factor 2 (same as factor 1 but for the left hand)
    % W : # of timesteps within the timeWindow array
    % R : # of repetitions for each condition set
    
    % Inputs: 
    % features : TxN feature matrix, T is number of time bins, N is # of channels
    % trialCodes: Kx2 array of factor conditions for each trial (K),  column 1 is Factor 1 conditions, column 2 is Factor 2 conditions 
    % eventIdx : Kx1 array of event indices (go cues) where a window (timeWindow) relative to these event indices will be extracted 
    % timeWindow : 1x2 array of timesteps relative to the eventIdx that will be extracted, first element is the start index, second element is the end index

    N = size(features,2);                           % number of features
    T = length(timeWindow(1):timeWindow(2));        % number of time steps in a trial
    nFactors = size(trialCodes,2);                  % number of factors

    % find unique trial codes
    [codeList,~,muxCodes] = unique(trialCodes, 'rows');
    maxRep = max(hist(muxCodes, max(muxCodes)));
    nCodes = size(codeList,1);
    nCons = zeros(nFactors, 1);
    for f=1:nFactors
        nCons(f) = length(unique(trialCodes(:,f)));
    end
    
    % initialize tensors 
    matrixSize_singleTrial = [N, nCons', T, maxRep];
    matrixSize_numTrials = [N, nCons'];
    featureVals = nan(matrixSize_singleTrial); 

    % extract appropriate windows for each trial and tile into the respective factors
    for codeIdx = 1:nCodes
        trlIdx = find(muxCodes==codeIdx);
        indOp = '(:';
        for f=1:nFactors
            indOp = [indOp, ',' num2str(codeList(codeIdx,f))];
        end
        indOp = [indOp, ')'];
        eval(['trialNum' indOp ' = length(trlIdx);']);
        
        indOp_neural = '(:';
        for f=1:nFactors
            indOp_neural = [indOp_neural, ',' num2str(codeList(codeIdx,f))];
        end
        indOp_neural = [indOp_neural, ',:,e)'];
        
        for e = 1:length(trlIdx)
            loopIdx = (eventIdx(trlIdx(e))+timeWindow(1)):(eventIdx(trlIdx(e))+timeWindow(2));
            if loopIdx(end)>size(features,1) || loopIdx(1)<1
                eval(['trialNum' indOp ' = trialNum' indOp ' - 1;']);
                continue;
            end
            
            eval(['featureVals' indOp_neural ' = features(loopIdx,:)'';']);
        end
    end

end
