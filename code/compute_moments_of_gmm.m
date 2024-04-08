function [mean_val,cov_val]=compute_moments_of_gmm(model)
    weights = model{1};
    mean_vector = model{2};
    variance_vector = model{3};
    
    mean_val = sum(weights.*mean_vector);
    num_of_comp=length(weights);
    
    size_of_mean = size(mean_vector);
    cov_val = zeros([size_of_mean(2),size_of_mean(2)]);
    
    for i=1:num_of_comp
        cov_val = cov_val+(weights(i).*(reshape(variance_vector(i,:,:),size(cov_val))+mean_vector(i,:)'*mean_vector(i,:)));
    end
    cov_val = cov_val-mean_val'*mean_val;

end