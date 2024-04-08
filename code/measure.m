function measurement=measure(state,measurement_model)
% rng(rnd_seed);

measurement_coef = measurement_model{1};
measurement_noise_mean = measurement_model{2};
measurement_noise_var = measurement_model{3};
measurement = normrnd(state.*measurement_coef+measurement_noise_mean,sqrt(measurement_noise_var));
end