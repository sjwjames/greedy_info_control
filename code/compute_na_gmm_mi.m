function MI=compute_na_gmm_mi(gmm_model,x1,x2)
    gmm_weights = gmm_model{1};
    gmm_means = gmm_model{2};
    gmm_covs = gmm_model{3};
    n_of_components = length(gmm_weights);
    [X1,X2] = meshgrid(x1,x2);
    delta_x = x1(2)-x1(1);
    delta_y = x2(2)-x2(1);
    X = [X1(:) X2(:)];
    p_gm = gmdistribution(gmm_means,gmm_covs,gmm_weights);
    original_joint_pdf = pdf(p_gm,X);
    original_x_means = gmm_means(:,1);
    original_y_means = gmm_means(:,2);
    original_x_vars = reshape(gmm_covs(1,1,:),[n_of_components,1]);
    original_y_vars = reshape(gmm_covs(2,2,:),[n_of_components,1]);

    original_x_pdf = gmm_weights*normpdf(x1,original_x_means,sqrt(original_x_vars));
    original_y_pdf = gmm_weights*normpdf(x2,original_y_means,sqrt(original_y_vars));

    non_zero_original_joint = original_joint_pdf(original_joint_pdf~=0);
    non_zero_original_x = original_x_pdf(original_x_pdf~=0);
    non_zero_original_y = original_y_pdf(original_y_pdf~=0);

    MI = -sum(non_zero_original_x.*log(non_zero_original_x).*delta_x)-sum(non_zero_original_y.*log(non_zero_original_y).*delta_y)+sum(non_zero_original_joint.*log(non_zero_original_joint).*delta_x.*delta_y);
end