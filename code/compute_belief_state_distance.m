function distance=compute_belief_state_distance(b1,b2,K)
    b1_weights = b1{1};
    N = length(b1_weights);
    b2_weights = b2{1};
    b1_samples = b1{2};
    b2_samples = b2{2};

%     KNN KL estimator
    if K>0
        b1_distance_table = b1{3};
        b1_to_b2_distance_table = abs(repmat(b1_samples,[1,N])-b2_samples');
        b1_k_nearest_distances=mink(b1_distance_table,K+1,2);
        b1_b2_k_nearest_distances=mink(b1_to_b2_distance_table,K,2);
        rho = sqrt(sum(b1_k_nearest_distances.^2,2));
        nu = sqrt(sum(b1_b2_k_nearest_distances.^2,2));
        distance = sum(log(nu./rho))/N+log(N/(N-1));
    else
        N = length(b1_weights);
        b1_resampled_cnts = mnrnd(N,b1_weights);
        b2_resampled_cnts = mnrnd(N,b2_weights);
        b1_samples  = repelem(b1_samples,b1_resampled_cnts);
        b2_samples  = repelem(b2_samples,b2_resampled_cnts);
        
        min_sample = min(min(b1_samples),min(b2_samples));
        max_sample = max(max(b1_samples),max(b2_samples));
        shared_X = linspace(min_sample,max_sample,1000);

        kde_estimate = ksdensity(b1_samples,shared_X);
        
    
        kde_estimate2 = ksdensity(b2_samples,shared_X);
        
    
        combined_dist = (kde_estimate + kde_estimate2).*0.5;
        delta_x = shared_X(2)-shared_X(1);
        
%         kde_estimate = kde_estimate./sum(kde_estimate);
%         kde_estimate2 = kde_estimate2./sum(kde_estimate2);
%         combined_dist = combined_dist./sum(combined_dist);
        
        non_zero1 = find(kde_estimate~=0);
        non_zero2 = find(kde_estimate2~=0);
        distance = 0.5*sum(kde_estimate(non_zero1).*(log(kde_estimate(non_zero1))-log(combined_dist(non_zero1))))+0.5*sum(kde_estimate2(non_zero2).*(log(kde_estimate2(non_zero2))-log(combined_dist(non_zero2))));
        distance = distance*delta_x;
        if isnan(distance)
            distance = log(2);
        end
    end
    


end


function [p1,p2]=compute_portion_of_samples(b1_samples,b2_samples)
    overlapped_min = max(min(b1_samples),min(b2_samples));
    overlapped_max = min(max(b1_samples),max(b2_samples));
    p1 = length(b1_samples(b1_samples>=overlapped_min & b1_samples<=overlapped_max))/length(b1_samples);
    p2 = length(b2_samples(b2_samples>=overlapped_min & b2_samples<=overlapped_max))/length(b2_samples);
end