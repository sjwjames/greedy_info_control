function [state_trans_probs,states]=discretize_transition_model(discretized_states,model)
     weights_vector = model{1};
    coefficients_vector = model{2};
    mean_vector = model{3};
    var_vector = model{4};
    
    num_of_comp = length(weights_vector);
    num_of_bins = length(discretized_states);

    last_states = discretized_states;
    current_states = repelem(discretized_states,num_of_comp*num_of_bins);
    states = discretized_states;
%     state_after_transition = normrnd((last_states'*coefficients_vector'+mean_vector'),repmat(sqrt(var_vector'),[num_of_bins,1]));
%     state_after_transition = sum(state_after_transition.*weights_vector',2);
%     state_range = [min(state_after_transition),max(state_after_transition)];
%     states = linspace(state_range(1),state_range(2),num_of_bins);
%     current_states = repelem(states,num_of_comp*num_of_bins);

    mean_vec_new = (last_states'*coefficients_vector'+mean_vector')';
    mean_vec_new = repmat(mean_vec_new(:)',[1,num_of_bins]);

    var_vec_new = repmat(repmat(var_vector',[1,num_of_bins]),[1,num_of_bins]);
    weights_vec_new = repmat(repmat(weights_vector',[1,num_of_bins]),[1,num_of_bins]);
    cur_state_probs = normpdf(current_states,mean_vec_new,sqrt(var_vec_new)).*weights_vec_new;
    state_trans_probs = sum(reshape(cur_state_probs',[num_of_comp,num_of_bins*num_of_bins]),1);
    
    
end