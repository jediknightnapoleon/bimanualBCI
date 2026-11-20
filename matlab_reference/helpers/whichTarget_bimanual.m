% Copied verbatim from tuningAnalysis.m (upstream d-r-deo/bimanualBCI) so the
% function can be called from a script under both MATLAB and Octave.

function TargID = whichTarget_bimanual(Targets, X)
    % X : Tx2 array of x and y-coordinates (position vectors) for each trial, T is the number of trials 
    % Targets is the target map as described on line 31
    % This function computes the angle (relative to origin) of each 2D position vector with the x -axis
    % Angles will range from 0 - 359 deg, and will return the associated target ID as governed by the target mapping 
    % TargID is returned, which is an array of the Target IDs for each position vector
    
    TargID = [];
    for i = 1 : size(X,1)
        % compute the angle for the current position vector
        theta = floor(angleFromXaxis(X(i,1), X(i,2)));
    
        if (X(i,1) == 0 ) || (X(i,2) == 0) % catch any position vectors without a x- or y-dimension
            TargID(i) = NaN;
        else
            for j = 1 : size(Targets, 2)
                if ismember(theta, Targets(:,j))
                    TargID(i) = j; % check the target map for which wedge the current theta resides
                end
            end
        end
    end
end
