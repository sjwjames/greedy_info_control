function [joint_model] = directly_compute_moment_matching(transition_model,measurement_model,approxmiated_mu,approximated_var)
    weights_vector = transition_model{1};
    coefficients_vector = transition_model{2};
    mu_vector = transition_model{3};
    var_vector = transition_model{4};
    
    num_of_components = length(weights_vector);
    
    
    H=measurement_model{1};
    measurement_noise_mean = measurement_model{2};
    measurement_noise_var = measurement_model{3};
    
    
    marginal_components_means=zeros([num_of_components,1]);
    marginal_components_vars=zeros([num_of_components,1]);
    for i=1:num_of_components
        marginal_components_means(i)=coefficients_vector(i)*approxmiated_mu+mu_vector(i);
        marginal_components_vars(i)=var_vector(i)+coefficients_vector(i)*approximated_var*coefficients_vector(i)';
    end
    
    joint_component_means = zeros([num_of_components,2]);
    joint_component_covs = zeros([num_of_components,2,2]);
    
    for i=1:num_of_components
        joint_component_means(i,:)=[marginal_components_means(i),H*marginal_components_means(i)+measurement_noise_mean];
        joint_component_covs(i,:,:)=[marginal_components_vars(i),marginal_components_vars(i)*H';marginal_components_vars(i)*H,measurement_noise_var+H*marginal_components_vars(i)*H'];
    end
    
    joint_model = {weights_vector,joint_component_means,joint_component_covs};





end