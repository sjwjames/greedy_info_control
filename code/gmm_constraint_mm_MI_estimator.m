function [pred_joint_dists,MI]=gmm_constraint_mm_MI_estimator(last_message,transition_model,measurement_model)
    last_weights = last_message{1};
    last_means = last_message{2};
    last_vars = last_message{3};
    num_of_comp = length(transition_model{1});
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    model_weights = transition_model{1};
    model_coef = transition_model{2};
    model_means = transition_model{3};
    model_vars = transition_model{4};
%    state marginal means
    m_ij = model_coef*last_means'+model_means;
 %    measurement marginal means
    M = H*sum(m_ij.*last_weights'.*model_weights,'all')+b;
    %    state marginal variance
    P_ij = model_coef*last_vars'.*model_coef+model_vars;
    %    measurement marginal variance
    P = sum((R+H.*P_ij.*H'+(H.*m_ij+b).*(H.*m_ij+b)).*last_weights'.*model_weights,'all')-M*M';
    MI = 0;
    joint_covs = [];
    joint_means = [];
    v_s = [];
    m_s = [];
    for j = 1:num_of_comp
        m_i = model_coef(j)*last_means+model_means(j);
        P_i = model_coef(j)*last_vars*model_coef(j)+model_vars(j);
        %             marginal moments of states when s=j
        m_j = dot(m_i,last_weights);
        m_s = [m_s;m_j];
        v_j = dot(P_i+m_i.*m_i,last_weights)-m_j*m_j;
        v_s = [v_s;v_j];
        %             marginal moments of measurements when s=j
        M_tilde_j =  dot(m_i.*H+b,last_weights);
        V_tilde_j = dot(R+P_i.*H^2+(m_i.*H+b).*(m_i.*H+b),last_weights)-M_tilde_j*M_tilde_j;
        cov_xy = dot(P_i.*H+m_i.*(m_i.*H+b),last_weights)-m_j.*M_tilde_j;
        F_j = (cov_xy+m_j*M_tilde_j)/(V_tilde_j+M_tilde_j^2);
        M_j = v_j+m_j^2-(cov_xy+m_j*M_tilde_j)^2/(V_tilde_j+M_tilde_j^2);
    
    
        MI = MI+0.5*model_weights(j)*log(det(M_j+F_j*P*F_j')/det(M_j));
        joint_covs(:,:,j)=[M_j+F_j*P*F_j',F_j*P;P*F_j',P];
        joint_means(j,:)=[m_j,M_j];
    end
    
    pred_joint_dists = {model_weights,joint_means,joint_covs};

    
end