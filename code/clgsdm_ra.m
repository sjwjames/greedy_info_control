function [state_dist_at_steps,discretized_states,information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,num_of_bins,T,K,state_measurements,prest_decisions,discretized_states,discretized_measurements,discretized_measurement_dist)
    states = state_measurements(:,1);
    measurements = state_measurements(:,2);
    information = zeros([1,T]);
    
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
%     hard-coded for the range
%   todo: dynamically compute the range
%     state_range = [min(states)-6,max(states)+6];
%     discretized_states = linspace(state_range(1),state_range(2),num_of_bins);
    delta_state = discretized_states(2)-discretized_states(1);
    delta_measurement = discretized_measurements(2)-discretized_measurements(1);
    state_dist = ones([1,num_of_bins]);
    initial_mu = initial_model{1};
    initial_var = initial_model{2};
    for i=1:num_of_bins
        state_dist(i)=normpdf(discretized_states(i),initial_mu,sqrt(initial_var));
    end
    
    state_dist_at_steps = zeros([T,num_of_bins]);
%     state_dist = state_dist./sum(state_dist)./delta_state;

    discretized_trans_dist_per_model = {};
    for d=1:K
        model = transition_models{d};
        [state_trans_probs,current_states] = discretize_transition_model(discretized_states,model);
        discretized_trans_dist_per_model{d}=state_trans_probs;
    end
    measurement_mean_vec = discretized_states.*H+b;
    measurement_var_vec = repmat(R,[1,num_of_bins]);
    measurement_dist_per_step = {};

    for t=1:T
       decision = prest_decisions(t);
        
       state_trans_probs = discretized_trans_dist_per_model{decision};
       [pred_state_dist]=compute_discretized_predictive(state_dist,state_trans_probs,num_of_bins,delta_state);
       
        
%         MI estimation
%        state_measuremen_joint_dist = discretized_measurement_dist.*pred_state_dist;
%        measurement_marginal = sum(state_measuremen_joint_dist,2).*delta_state;
%        info_gain = sum(state_measuremen_joint_dist.*log(state_measuremen_joint_dist./pred_state_dist./measurement_marginal').*delta_measurement.*delta_state,"all");
%        if t==1
%            information(t) = info_gain;
%        else
%            information(t) = information(t-1) + info_gain;
%        end
%        current_measurement = measurements(t);
%        measurement_dist_per_step{t} = normpdf(current_measurement,measurement_mean_vec,sqrt(measurement_var_vec));
%        temp_dist=pred_state_dist.*measurement_dist_per_step{t};
%        posterior_state_dist = temp_dist./sum(temp_dist)./delta_state;
%        state_dist = posterior_state_dist;
%        state_dist_at_steps(t,:)=posterior_state_dist;
       
%         entropy reduction estimation

       current_measurement = measurements(t);

       log_measurement_dist = log_normpdf(current_measurement,measurement_mean_vec,sqrt(measurement_var_vec));
       log_measurement_dist = log_measurement_dist-max(log_measurement_dist);
       measurement_dist_per_step{t} = exp(log_measurement_dist);
       log_joint_dist = log_measurement_dist+log(pred_state_dist);
       log_joint_dist = log_joint_dist-max(log_joint_dist);
%        state_dist = exp(log_state_dist);
%        measurement_marginal = sum(state_dist)*delta_state;
%        log_posterior_state_dist = log_state_dist - log(measurement_marginal);
%        log_posterior_state_dist = log_posterior_state_dist-max(log_posterior_state_dist);
       posterior_state_dist = exp(log_joint_dist)./sum(exp(log_joint_dist))./delta_state;

       info_gain =  sum(-pred_state_dist(pred_state_dist~=0).*log(pred_state_dist(pred_state_dist~=0)).*delta_state)-sum(-posterior_state_dist(posterior_state_dist~=0).*log(posterior_state_dist(posterior_state_dist~=0)).*delta_state);
       if t==1
           information(t) = info_gain;
       else
           information(t) = information(t-1) + info_gain;
       end

       state_dist = posterior_state_dist;
       state_dist_at_steps(t,:)=posterior_state_dist;
       
    %    figure,
    %    plot(discretized_states,state_dist');
    %    title("approximated filter distribution of state at time "+string(t));
    %    saveas(gcf,'../experiment_results/greedy ep/synthetic preset decisions/riemann sum approximation/'+"approximated filter distrribution of state at time "+string(t)+'.png');
    end
    
%     info =  sum(-state_dist(state_dist~=0).*log(state_dist(state_dist~=0)).*delta_state);
%     information(T) = info;
% 
%     backward_messages = ones([1,num_of_bins]);
%     for t=T-1:-1:1
%         likelihood_probs = repelem(measurement_dist_per_step{t+1},num_of_bins);
%         d = prest_decisions(t+1);
%         state_trans_probs = discretized_trans_dist_per_model{d};
%         backward_message_vec = repelem(backward_messages,num_of_bins);  
%         backward_message_vec = backward_message_vec.*state_trans_probs.*likelihood_probs;
%         backward_messages = sum(reshape(backward_message_vec.*delta_state,[num_of_bins,num_of_bins]),2)';
%         smoothed_at_t = backward_messages.*state_dist_at_steps(t,:);
%         smoothed_at_t = smoothed_at_t./sum(smoothed_at_t)./delta_state;
%         state_dist_at_steps(t,:) = smoothed_at_t;
% 
%     end

    
end

