function [mi]=particle_MI_estimation(num_of_samples,num_of_comp,samples,sample_weights,transition_model,measurement_model,samples_of_last)
        weights = transition_model{1};
        coefficients = transition_model{2};
        means = transition_model{3};
        variance = transition_model{4};
        
        H = measurement_model{1};
        b = measurement_model{2};
        R = measurement_model{3};
        
        state_marginal_probs = zeros([1,num_of_samples]);
        measurment_marginal_probs = zeros([1,num_of_samples]);
        joint_probs = zeros([1,num_of_samples]);

        predictive_variance = repmat(variance,[1,num_of_samples-1]);
        measurement_marginal_variance = H*predictive_variance*H'+R;
        for s=1:num_of_samples
            state_sampled = samples(s,1);
            measurement_sampled = samples(s,2);
            indices = setdiff(1:num_of_samples,s);
            samples_excludes_s=samples_of_last(indices);
            sample_weights_excludes_s=sample_weights(indices);
            predictive_means = coefficients*(samples_excludes_s')+means;
            predictive_state_probs = zeros([num_of_comp,num_of_samples-1]);
            measurement_marginal_mean = predictive_means.*H+b;
            measurement_prob = zeros([num_of_comp,num_of_samples-1]);
            for n = 1:num_of_comp
                predictive_state_probs(n,:) = normpdf(state_sampled,predictive_means(n,:),sqrt(predictive_variance(n,:)));
                measurement_prob(n,:)=normpdf(measurement_sampled,measurement_marginal_mean(n,:),sqrt(measurement_marginal_variance(n,:)));
            end
            predictive_state_probs = sum(predictive_state_probs.*weights,1);
            state_marginal_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs);
            joint_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs.*normpdf(measurement_sampled,H*state_sampled+b,sqrt(R)));
            measurement_prob = sum(measurement_prob.*weights,1);
            measurment_marginal_probs(s) = dot(sample_weights_excludes_s,measurement_prob);
        end
        state_marginal_probs(state_marginal_probs==0)=1;
        measurment_marginal_probs(measurment_marginal_probs==0)=1;
        joint_probs(joint_probs==0)=1;
        predictive_state_entropy = -dot(sample_weights,log(state_marginal_probs));
        predictive_measuerment_entropy = -dot(sample_weights,log(measurment_marginal_probs));
        predictive_joint_entropy = -dot(sample_weights,log(joint_probs));
        mi = predictive_state_entropy+predictive_measuerment_entropy-predictive_joint_entropy;
end