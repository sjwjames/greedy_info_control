function  [next_state,sampled_component] = dynamic_transition(current_state,decision,transition_model)
% rng(rnd_seed);
model = transition_model{decision};
weights_vector = model{1};
coefficients_vector = model{2};
mu_vector = model{3};
var_vector = model{4};



assignment = mnrnd(1,weights_vector);
weight_of_assignment = weights_vector(assignment==1);
coef = coefficients_vector(assignment==1);
mu = mu_vector(assignment==1);
var = var_vector(assignment==1);
sampled_component = {weight_of_assignment,coef,mu,var};
next_state = normrnd(coef*current_state+mu,sqrt(var));

end