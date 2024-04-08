function [forward_message_new]=update_after_measurement_multi(model,measurement_model,measurement,last_forward_message,dim)
        posterior_model = compute_posterior_multi(model,measurement_model,measurement,last_forward_message,dim);
        [posterior_mean,posterior_var] = compute_moments_of_gmm_multi(posterior_model,dim);
        forward_message_new = {posterior_mean,posterior_var};
end