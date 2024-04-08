% Estimate the maximum mean discrepancy between two belief states
function [distance,testStat,thresh]=mmd_estimation(b1,b2)
%     resampling to make sure each of them are from the target distribution
    b1_weights = b1{1};
    N = length(b1_weights);
    b2_weights = b2{1};
    b1_samples = b1{2};
    b2_samples = b2{2};

    b1_sampled_cnts = mnrnd(N,b1_weights);
    b1_resampled_states  = repelem(b1_samples,b1_sampled_cnts);
    b2_sampled_cnts = mnrnd(N,b2_weights);
    b2_resampled_states  = repelem(b2_samples,b2_sampled_cnts);
    
%     b1_resampled_states = b1_samples;
%     b2_resampled_states = b2_samples;
    
%     by default it is RBF kernel, set variance as 1
%     sigma = 1;
% 
%     XX = exp((-(b1_resampled_states'-b1_resampled_states).^2)./(2*sigma^2));
%     YY = exp((-(b2_resampled_states'-b2_resampled_states).^2)./(2*sigma^2));
%     XY = exp((-(b1_resampled_states'-b2_resampled_states).^2)./(2*sigma^2));
%     distance = (sum(XX,"all")-sum(diag(XX)))/N/(N-1)+(sum(YY,"all")-sum(diag(YY)))/N/(N-1)-2*mean(XY,"all");
    
    alpha = 0.05;
    params.numEigs = -1;
    params.numNullSamp = 100;
    params.plotEigs = false;
    params.sig = -1;
    [testStat,thresh,params] = mmdTestGamma(b1_resampled_states,b2_resampled_states,alpha,params);
    distance = testStat/N;
end