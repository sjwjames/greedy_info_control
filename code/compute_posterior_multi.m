function [posterior_model]=compute_posterior_multi(model,measurement_model,measurement,last_forward_message,dim)
        weights_vector = model{1};
        coefficients_vector = model{2};
        mu_vector = model{3};
        covs = model{4};
        H = measurement_model{1};
        b = measurement_model{2}; 
        R = measurement_model{3};
        last_mean = last_forward_message{1};
        last_cov = last_forward_message{2};
        num_of_components = length(weights_vector);
        
        posterior_components_means=zeros([num_of_components,dim]);
        posterior_components_covs=zeros([dim,dim,num_of_components]);
        for i=1:num_of_components
            marginal_pre = inv(covs(:,:,i)+coefficients_vector(:,:,i)*last_cov*coefficients_vector(:,:,i)');
            marginal_mean = last_mean*coefficients_vector(:,:,i)'+mu_vector(i,:);
            posterior_components_covs(:,:,i)=inv(marginal_pre+H'/R*H);
            posterior_components_means(i,:) = ((measurement-b)*(H'/R)'+marginal_mean*marginal_pre')*posterior_components_covs(:,:,i)';
            likelihood_mean = marginal_mean*H'+b;
            likelihood_cov = H/marginal_pre*H'+R;
            weights_vector(i) = mvnpdf(measurement,likelihood_mean,likelihood_cov);
        end
        weights_vector = weights_vector./sum(weights_vector);
        posterior_model = {weights_vector,posterior_components_means,posterior_components_covs};

end