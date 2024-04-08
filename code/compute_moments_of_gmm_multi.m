function [mean_val,cov_val] = compute_moments_of_gmm_multi(model,dim)
    weights = model{1};
    % number of components x dim
    mean_vector = model{2};
    % dim x dim x number of components
    covs = model{3};
    
    mean_val = weights'*mean_vector;
    num_of_comp=length(weights);
    
    cov_val = zeros([dim,dim]);
    
    for i=1:num_of_comp
        cov_val = cov_val+(weights(i).*(covs(:,:,i)+mean_vector(i,:)'*mean_vector(i,:)));
    end
    cov_val = cov_val-mean_val'*mean_val;
end