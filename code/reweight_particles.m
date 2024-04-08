function [samples,weights_new]=reweight_particles(num_of_samples,num_of_comp,sample_weights,samples_of_last,transition_model,measurement_model,measurement,effective_sample_threshold)
    weights = transition_model{1};
    coefficients = transition_model{2};
    means = transition_model{3};
    variance = transition_model{4};
        
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    samples = zeros([num_of_samples,2]);
    weights_new = zeros([1,num_of_samples]);
    
    states_sampled = zeros([num_of_samples,num_of_comp]);
    comp_variances = (1./(1./variance+H'*(1./R)*H))';
    states_of_last = samples_of_last(:,1);
    weight_mean = (states_of_last.*coefficients'+means').*H+b;
    weight_var = repmat((H*variance*H'+R)',[num_of_samples,1]);
    posterior_means = ((states_of_last.*coefficients'+means').*(1./variance')+H'/R*(measurement-b)).*comp_variances;
    weights_posterior = zeros([num_of_samples,num_of_comp]);
    for c=1:num_of_comp
        weights_posterior(:,c) = normpdf(measurement,weight_mean(:,c),sqrt(weight_var(:,c)));
    end

    posterior_weights = weights_posterior./sum(weights_posterior,2);
    
    for s =1:num_of_samples
        assignment = mnrnd(1,posterior_weights(s,:));
        mean_val = posterior_means(s,assignment==1);
        var_val = comp_variances(assignment==1);
        samples(s,1) = normrnd(mean_val,sqrt(var_val));
    end
    samples(:,2) = normrnd(samples(:,1).*H+b,sqrt(R));
%     tofix weight computation
%     weights_new = sample_weights*dot(repmat(weights',[num_of_samples,1]),posterior_weights);
    for s=1:num_of_samples
        weights_new(s) = sample_weights(s)*dot(weights',posterior_weights(s,:));
    end
%     for s=1:num_of_samples
%         state_of_last_step = samples_of_last(s);
%         gmm_posterior_variance = 1./(1./variance+H'*(1./R)*H);
%         gmm_posterior_mean =  gmm_posterior_variance.*(H'/R*(measurement-b)+1./variance.*(coefficients.*state_of_last_step+means));
%         sigma = reshape(sqrt(gmm_posterior_variance),[1,1,num_of_comp]);
%         weight_mean = H*(coefficients.*state_of_last_step+means)+b;
%         weight_var = H*variance*H'+R;
%         gmm_posterior_weights = normpdf(measurement,weight_mean,sqrt(weight_var));
%         gmm_posterior_weights = gmm_posterior_weights./sum(gmm_posterior_weights);
%         gmm = gmdistribution(gmm_posterior_mean,sigma,gmm_posterior_weights');
%         state_sampled= random(gmm,1);
%         samples(s,1) = state_sampled;
%         measurement_sampled = normrnd(H*state_sampled+b,sqrt(R));
%         samples(s,2) = measurement_sampled;
%         weights_new(s) = sample_weights(s)*dot(weights,normpdf(measurement,weight_mean,sqrt(weight_var)));
%     end
    weights_new = weights_new./sum(weights_new);
    sample_efficiency = 1/sum(weights_new.^2);
    if sample_efficiency<effective_sample_threshold
         sample_copies = mnrnd(num_of_samples,weights_new);
         samples(:,1) = repelem(samples(:,1),sample_copies);
         samples(:,2) = repelem(samples(:,2),sample_copies);
         weights_new = ones([1,num_of_samples]).*(1/num_of_samples);
    end
   
end