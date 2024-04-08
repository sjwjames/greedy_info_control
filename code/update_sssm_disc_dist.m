function [forward_discretized_message,entropy_reduction_sum] = update_sssm_disc_dist(selected_augmented_dist,measurement,measurement_model,discretized_states,K,N,delta_X,delta_Y)
    forward_discretized_message = zeros([K,N]);
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    likelihood = normpdf(repmat(measurement,[1,N]),H.*discretized_states+b,sqrt(R))';
    
    marginal_likelihoods = zeros([K,1]);
    predictive_marginal = zeros([K,N]);
    for j=1:K
        xy_dist = selected_augmented_dist{j};
        x_dist = sum(xy_dist,2).*delta_Y;
        predictive_marginal(j,:) = x_dist;
        joint_dist = x_dist.*likelihood;
        marginal_likelihood_with_s = sum(joint_dist).*delta_X;
        marginal_likelihoods(j) = marginal_likelihood_with_s;

        forward_discretized_message(j,:) = joint_dist';
        
    end
    
    if sum(marginal_likelihoods)~=0
        forward_discretized_message = forward_discretized_message./sum(marginal_likelihoods);
    else
        disp(1)
    end
    weighted_posterior = sum(forward_discretized_message,1);
    predictive_marginal_sum = sum(predictive_marginal,1);
    non_zero_predictive_marginal = predictive_marginal_sum(predictive_marginal_sum~=0);
    non_zero_posterior = weighted_posterior(weighted_posterior~=0);
    entropy_reduction_sum = -sum(non_zero_predictive_marginal.*log(non_zero_predictive_marginal).*delta_X)+sum(non_zero_posterior.*log(non_zero_posterior).*delta_X);
    
end