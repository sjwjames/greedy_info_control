% get the moments of measurement model in the CPOMDP active information
% acquisition variant
function moments=get_measurement_moments(z,x)
    moments = {z,0.5*abs(z-x)+0.05};
end