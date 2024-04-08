function [current_max_d]=particle_filter_decision(num_of_samples,num_of_comp,K,sample_weights,samples_of_last,transition_models,measurement_model,fixed_decision)
    current_max_d = 0;
    current_max = 0;
    for d = 1:K
        if fixed_decision==0||fixed_decision==d
             transition_model = transition_models{d};
     
        
            H = measurement_model{1};
            b = measurement_model{2};
            R = measurement_model{3};
            states_of_last = samples_of_last(:,1);
            samples = zeros([num_of_samples,2]);
            
    %         for s=1:num_of_samples
    %             samples(s,1) = dynamic_transition(states_of_last(s),d,transition_models);
    %         end
            samples(:,1) = dynamic_transition(states_of_last,d,transition_models);
            samples(:,2) = normrnd(samples(:,1).*H+b,sqrt(R));
    
    %         for s=1:num_of_samples
    %             last_state = states_of_last(s,1);
    %             gmm_mean =  coefficients.*last_state+means;
    %             gmm = gmdistribution(gmm_mean,sigma,weights');
    %             state_sampled= random(gmm,1);
    %             samples(s,1) = state_sampled;
    %             measurement_sampled = normrnd(H*state_sampled+b,sqrt(R));
    %             samples(s,2) = measurement_sampled;
    %         end
            mi=particle_MI_estimation(num_of_samples,num_of_comp,samples,sample_weights,transition_model,measurement_model,states_of_last);
            if mi>=current_max
                current_max = mi;
                current_max_d =d;
            end
        end
       
    end

end