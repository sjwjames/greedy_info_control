function [posterior_model]=compute_posterior(model,measurement_model,measurement,last_forward_message)
        weights_vector = model{1};
        coefficients_vector = model{2};
        mu_vector = model{3};
        var_vector = model{4};
        measurement_coef = measurement_model{1};
        measurement_noise_mean = measurement_model{2};
        measurement_noise_var = measurement_model{3};
        last_mean = last_forward_message{1};
        last_var = last_forward_message{2};
        num_of_components = length(weights_vector);
        
        posterior_components_means=zeros([num_of_components,1]);
        posterior_components_vars=zeros([num_of_components,1]);
        for i=1:num_of_components
            marginal_pre = 1/(var_vector(i)+coefficients_vector(i)*last_var*coefficients_vector(i)');
            marginal_mean = coefficients_vector(i)*last_mean+mu_vector(i);
            posterior_components_vars(i)=1/(marginal_pre+measurement_coef.^2/measurement_noise_var);
            posterior_components_means(i) = posterior_components_vars(i)*(measurement_coef/measurement_noise_var*(measurement-measurement_noise_mean)+marginal_pre*marginal_mean);
            weight_mean = measurement_coef*marginal_mean+measurement_noise_mean;
            weight_var = measurement_coef/marginal_pre*measurement_coef'+measurement_noise_var;
            weights_vector(i) = normpdf(measurement,weight_mean,sqrt(weight_var));
        end
        weights_vector = weights_vector./sum(weights_vector);
        posterior_model = {weights_vector,posterior_components_means,posterior_components_vars};

end