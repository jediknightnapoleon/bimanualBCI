% Copied verbatim from tuningAnalysis.m (upstream d-r-deo/bimanualBCI) so the
% function can be called from a script under both MATLAB and Octave.

function angle = angleFromXaxis(x,y)
% Computes the angle (in degrees) from the x-axis of the position vector [x y]
    u = [1 0 0];
    v = [x y 0];
    
    norm_u = norm(u);
    norm_v = norm(v);
    
    dotprod = dot(u,v);
    
    theta = acosd(dotprod/(norm_u*norm_v));
    
    angle = 0;
    if (x >= 0 && y >=0) || (x <=0 && y>=0) 
        angle = theta;
    else
        angle = 360-theta;
    end
    
    if (x==0) && (y == 0)
        angle = 0;
    end
end
