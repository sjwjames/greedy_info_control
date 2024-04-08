function [pred_state_dist]=compute_discretized_predictive(state_dist,state_trans_probs,num_of_bins,delta_state)
  last_state_dist_vec = repmat(state_dist,[1,num_of_bins]);
  cur_state_probs = state_trans_probs.*last_state_dist_vec;
  pred_state_dist = sum(reshape(cur_state_probs,[num_of_bins,num_of_bins]),1).*delta_state;
%   pred_state_dist = cur_state_probs./sum(cur_state_probs);
end