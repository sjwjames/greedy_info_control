function [forward_message_new]=update_after_measurement(model,measurement_model,measurement,last_forward_message)
        posterior_model = compute_posterior(model,measurement_model,measurement,last_forward_message);
        [posterior_mean,posterior_var] = compute_moments_of_gmm(posterior_model);
        forward_message_new = {posterior_mean,posterior_var};
end