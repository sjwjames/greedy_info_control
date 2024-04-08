function [model_new]=recompute_weight_after_measuring(last_mean,last_variance,model,measurement_coef,measurement_noise_mean,measurement_noise_var,measurement)
model_new = model;
weights_vector = model_new{1};
comp_len = length(weights_vector);
coefficients_vector = model_new{2};
mu_vector = model_new{3};
var_vector = model_new{4};
for i=1:comp_len
    coef = coefficients_vector(i);
    noise_mean = mu_vector(i);
    noise_var = var_vector(i);
    weight_mean = measurement_coef*(coef*last_mean+noise_mean)+measurement_noise_mean;
    weight_var = measurement_coef*(noise_var+coef*last_variance*coef')*measurement_coef'+measurement_noise_var;
    weights_vector(i)=normpdf(measurement,weight_mean,sqrt(weight_var));
end
weights_vector=weights_vector./sum(weights_vector,'all');
model_new{1}=weights_vector;

end
