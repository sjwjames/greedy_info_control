function measurement=measure_multi(state,measurement_model,rnd_seed)
%     rng(rnd_seed);
    
    measurement_coef = measurement_model{1};
    measurement_noise_mean = measurement_model{2};
    measurement_noise_cov = measurement_model{3};
    measurement = mvnrnd(state*measurement_coef+measurement_noise_mean,measurement_noise_cov);
end