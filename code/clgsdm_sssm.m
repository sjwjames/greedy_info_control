function [ADF_results,discretization_results,pf_results,state_check]=clgsdm_sssm(initial_model,sssm,measurement_model,T,K,D,dim,discretization_values,num_of_samples,fixed_setting)
    forward_messages = {};
    ADF_results = {};
    current_s = initial_model{1};
    current_x = initial_model{2};
    xs = zeros([1,T]);
    ss = zeros([1,T]);
    measurements = [];
    decisions = [];
    
%  discretization results and values
    N = discretization_values{1};
    discretized_states = discretization_values{2};
    discretized_measurements = discretization_values{3};
    discretized_sssm = discretization_values{4};
    discretized_measurement_model = discretization_values{5};
    initial_discretization_message = discretization_values{6};
    delta_X = discretized_states(2)-discretized_states(1);
    delta_Y = discretized_measurements(2)-discretized_measurements(1);
    discretization_forward_messages = {};
    discretization_forward_messages{1} = initial_discretization_message;
    discretization_current_s = initial_model{1};
    discretization_current_x = initial_model{2};
    discretization_xs = zeros([1,T]);
    discretization_ss = zeros([1,T]);
    discretization_measurements = [];
    discretization_decisions = [];
    discretization_MI = [];

%     from the initial model to construct the first forward message
    initial_weights = ones([K,1]).*(1/K);
    initial_means = repmat(initial_model{4},[K,1]);
    initial_vars = repmat(initial_model{5},[K,1]);
    forward_message = {initial_weights,initial_means,initial_vars};
    forward_messages{1} = forward_message;

%     PF
    sample_weights = ones([1,num_of_samples]).*(1/num_of_samples);
    samples_of_cates = ones([1,num_of_samples]).*current_s;
    samples_of_states = ones([1,num_of_samples]).*current_x;
    samples_of_means = ones([1,num_of_samples]).*initial_model{4};
    samples_of_vars = ones([1,num_of_samples]).*initial_model{5};
    initial_samples = {sample_weights,samples_of_cates,samples_of_means,samples_of_vars,samples_of_states};
    samples{1} = initial_samples;
    effective_sample_threshold = inf;
    pf_current_s = initial_model{1};
    pf_current_x = initial_model{2};
    pf_xs = zeros([1,T]);
    pf_ss = zeros([1,T]);
    pf_measurements = [];
    pf_decisions = [];
    ADF_results = {};
    discretization_results = {};
    pf_results = {};
    
    fixed_decisions = [];
    fixed_s = [];
    fixed_states = [];
    fixed_measurement = [];
    if ~isempty(fixed_setting)
        fixed_decisions = fixed_setting{1};
        fixed_s = fixed_setting{2};
        fixed_states = fixed_setting{3};
        fixed_measurement = fixed_setting{4};
    end
    
    for t=1:T
%         discretization method
        disp("t="+string(t));
        fixed_d = 0;
        if ~isempty(fixed_decisions)
            fixed_d = fixed_decisions(t);
        end
        [discretized_max_d,discretized_selected_augmented_dist]=make_discretization_decision(discretization_forward_messages{t},discretized_sssm,discretized_measurement_model,K,D,N,delta_X,delta_Y,fixed_d);
        discretization_decisions = [discretization_decisions;discretized_max_d];
        [d_s,d_x]=process_sssm_transition(discretization_current_s,discretization_current_x,discretized_max_d,sssm);
        d_measurement = measure(d_x,measurement_model);
        if ~isempty(fixed_s)&&~isempty(fixed_measurement)&&~isempty(fixed_states)
            d_s = fixed_s(t);
            d_x = fixed_states(t);
            d_measurement = fixed_measurement(t);
        end
        discretization_current_s = d_s;
        discretization_current_x = d_x;
        discretization_xs(t)=discretization_current_x;
        discretization_ss(t)=discretization_current_s;
        discretization_measurements = [discretization_measurements;d_measurement];
        [forward_discretized_message,entropy_reduction_sum] = update_sssm_disc_dist(discretized_selected_augmented_dist,d_measurement,measurement_model,discretized_states,K,N,delta_X,delta_Y);
        discretization_forward_messages{t+1} = forward_discretized_message;
        if t==1
            discretization_MI = [discretization_MI;entropy_reduction_sum];
        else
            discretization_MI = [discretization_MI;entropy_reduction_sum+discretization_MI(t-1)];
        end


        %      compute the augmented distribution
        [augmented_distributions,approximated_augmented_distributions,d,current_max]=make_sssm_decision_ADF(forward_message,sssm,measurement_model,K,D,dim,fixed_d);
        decisions = [decisions;d];
%     process the state to next step, states are not observable to the agent
        [s,x]=process_sssm_transition(current_s,current_x,d,sssm);
        measurement = measure(x,measurement_model);
        if ~isempty(fixed_s)&&~isempty(fixed_measurement)&&~isempty(fixed_states)
            s = fixed_s(t);
            x = fixed_states(t);
            measurement = fixed_measurement(t);
        end
        current_s = s;
        current_x = x;
        xs(t)=current_x;
        ss(t)=current_s;
        measurements = [measurements;measurement];
%      update the forward message based on the measurement
        updated_message = sssm_update_forward_message(augmented_distributions,d,measurement,measurement_model,K,dim);
        forward_messages{t+1} = updated_message;


%         pf
        [pf_max_d,new_sample_cates]=particle_filter_decision_sssm(samples{t},sssm,measurement_model,K,D,num_of_samples,fixed_d);
        pf_decisions = [pf_decisions;pf_max_d];
        [pf_s,pf_x]=process_sssm_transition(pf_current_s,pf_current_x,d,sssm);
        pf_measurement = measure(pf_x,measurement_model);
        if ~isempty(fixed_s)&&~isempty(fixed_measurement)&&~isempty(fixed_states)
            pf_s = fixed_s(t);
            pf_x = fixed_states(t);
            pf_measurement = fixed_measurement(t);
        end
        pf_current_s = pf_s;
        pf_current_x = pf_x;
        pf_xs(t)=pf_current_x;
        pf_ss(t)=pf_current_s;
        pf_measurements = [pf_measurements;pf_measurement];

        [new_samples]=reweight_particles(num_of_samples,new_sample_cates,pf_max_d,samples{t},sssm,measurement_model,pf_measurement,K,effective_sample_threshold);   
        samples{t+1} = new_samples;

        state_upper_check = pf_current_x<discretized_states(N)&&current_x<discretized_states(N)&&discretization_current_x<discretized_states(N);
        state_lower_check = pf_current_x>discretized_states(1)&&current_x>discretized_states(1)&&discretization_current_x>discretized_states(1);
        state_check = state_upper_check && state_lower_check;
        if ~state_check
            return
        end
    end
    ADF_results={decisions,xs,ss,measurements,forward_messages};
    discretization_results = {discretization_decisions,discretization_xs,discretization_ss,discretization_measurements,discretization_forward_messages,discretization_MI};
    pf_results = {pf_decisions,pf_xs,pf_ss,pf_measurements,samples};
end


function [max_d,selected_augmented_dist]=make_discretization_decision(forward_dist,discretized_sssm,discretized_measurement_model,K,D,N,delta_X,delta_Y,fixed_d)
    rhos = discretized_sssm{1};
    x_transitions = discretized_sssm{2};
    s_marginalized=rhos'*forward_dist;
    max_d = 0;
    current_max_MI = 0;
    selected_augmented_dist = [];
    for d=1:D
        if fixed_d == 0 || fixed_d==d
            joint_entropy=0;
            marginal_entropy = 0;
            y_dist = zeros([1,N]);
            joint_distribution_collection = {};
            for j=1:K
                predictive_marginal = sum(s_marginalized(j,:)'.*x_transitions{j}{d}.*delta_X,1);
                joint_distribution = predictive_marginal'.*discretized_measurement_model;
    
                non_zero_joint = joint_distribution(joint_distribution~=0);
                joint_entropy = joint_entropy-sum(non_zero_joint.*log(non_zero_joint).*delta_X.*delta_Y,'all');
                non_zero_marginal = predictive_marginal(predictive_marginal~=0);
                marginal_entropy = marginal_entropy-sum(non_zero_marginal.*log(non_zero_marginal).*delta_X,'all');
                y_dist = y_dist + sum(joint_distribution,1).*delta_X;
                joint_distribution_collection{j} = joint_distribution;
            end
            non_zero_likelihood = y_dist(y_dist~=0);
            marginal_likelihood_entropy = -sum(non_zero_likelihood.*log(non_zero_likelihood).*delta_Y,'all');
            MI = marginal_entropy+marginal_likelihood_entropy-joint_entropy;
            if MI>current_max_MI
                current_max_MI = MI;
                max_d = d;
                selected_augmented_dist = joint_distribution_collection;
            end
        end
        
    end

end

function [augmented_distributions,approximated_augmented_distributions,current_max_d,current_max]=make_sssm_decision_ADF(forward_message,sssm,measurement_model,K,D,dim,fixed_d)
    weights = forward_message{1};
    mus = forward_message{2};
    sigmas = forward_message{3};
    augmented_distributions = {};
    rhos = sssm{1};
    weights_unnorm = rhos.* weights;
    weights_overall = sum(weights_unnorm,1)./sum(weights_unnorm,"all");
    weights_new = weights_unnorm./sum(weights_unnorm,1);
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    approximated_augmented_distributions = {};
    current_max_d = 0;
    current_max = 0;
    for d=1:D
        if fixed_d==0||fixed_d==d
            y_vars = [];
            y_means = [];
            approx_weights = [];
            approx_means = zeros([K,2,dim]);
            approx_covs =  zeros([K,dim*2,dim*2]);
            model_means = zeros([K,1]);
            model_coefs = zeros([K,1]);
            model_weights = zeros([K,1]);
            model_vars = zeros([K,1]);
            for j=1:K 
                params = sssm{2}{j}{d};
                coefficient = params{1};
                state_noise_mean = params{2};
                state_noise_variance = params{3};
                model_vars(j) = state_noise_variance;
                model_means(j) = state_noise_mean;
                model_coefs(j) = coefficient;
                model_weights(j) = sum(rhos(:,j));
    
                Ms = zeros([K,2,dim]);
                Lamda_invs = zeros([K,dim*2,dim*2]);
                for i=1:K
                    mu = mus(i);
                    sigma = sigmas(i);
                    mu_bar = coefficient*mu+state_noise_mean;
                    sigma_bar = state_noise_variance+coefficient*sigma*coefficient';
                    Ms(i,:,:) = [mu_bar,H*mu_bar+b];
                    Lamda_invs(i,:,:) = [sigma_bar,sigma_bar*H';H*sigma_bar,R+H*sigma_bar*H'];
                    y_vars = [y_vars;R+H*sigma_bar*H'];
                    y_means = [y_means;H*mu_bar+b];
                end
                augmented_distributions{d}{j}={weights_new(:,j),Ms,Lamda_invs,weights_overall(j)};
                [approx_mean_val,approx_cov_val]=compute_moments_of_gmm(augmented_distributions{d}{j});
                approx_weights = [approx_weights;weights_overall(j)];
                approx_means(j,:,:) = approx_mean_val;
                approx_covs(j,:,:) = approx_cov_val;
            end
            
            [approx_pred_joint_dists,MI]=gmm_constraint_mm_MI_estimator(forward_message,{model_weights,model_coefs,model_means,model_vars},measurement_model);
            [mean_val,approx_y_var]=compute_moments_of_gmm({reshape(weights_unnorm,[K^2,1]),y_means,y_vars});
            approximated_augmented_distributions{d} = {approx_weights,approx_means,approx_covs,approx_y_var};
            if MI>current_max
              current_max=MI;
              current_max_d = d;
           end

        end
    end
end

% function [current_max_d]=make_sssm_decision_ADF(approximated_augmented_dists,D,K,dim)
%     current_max_d = 0;
%     current_max = 0;
%     for d=1:D
%         current_info_gain = 0;
%         approximated_augmented_dist = approximated_augmented_dists{d};
%         weights = approximated_augmented_dist{1};
%         covs = approximated_augmented_dist{3};
%         p = approximated_augmented_dist{4};
%         for j=1:K
%             cov = reshape(covs(j,:,:),[dim*2,dim*2]);
%             precision = inv(cov);
%             cov_marginal_x = cov(1:dim,1:dim);
%             p_h = inv(precision(1:dim,1:dim));
%             fpf = cov_marginal_x - p_h;
%             if dim~=1
%                 A_h=chol(fpf)';
%                 L=chol(p)';
%                 F_h = A_h/L;
%                 current_info_gain = current_info_gain + 0.5*weights(j)*log(det(p_h+F_h*p*F_h')/det(p_h));
%             else
%                 A_h=fpf;
%                 L=p;
%                 F_h = A_h/L;
%                 current_info_gain = current_info_gain + 0.5*weights(j)*log(det(p_h+F_h*p*F_h')/det(p_h));
%             end
%         end
% %         [joint_mean,joint_cov]=compute_moments_of_gmm(approximated_augmented_dists{d});
% %         marginal_var = joint_cov(1:dim,1:dim);
% %        likelihood_var = joint_cov(dim+1:2*dim,dim+1:2*dim);
% %        current_info_gain = 0.5*log(det(marginal_var))+ 0.5*log(det(likelihood_var))-0.5*log(det(joint_cov));
% 
% %        hardcoded the number of samples now
%        MCMI = compute_augmented_dist_MI_MC(approximated_augmented_dist,6000);
% %        disp(abs(MCMI-current_info_gain))
%        if MCMI>current_max
%           current_max=MCMI;
%           current_max_d = d;
%        end
%     end
% end

function [forward_message]=sssm_update_forward_message(augmented_distributions,decision,measurement,measurement_model,K,dim)
    selected_model = augmented_distributions{decision};
    forward_means = zeros([K,1,dim]);
    forward_covs = zeros([K,dim,dim]);
    forward_weights = [];
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    for j=1:K
         weights = selected_model{j}{1};
         means = zeros([K,1,dim]);
         covs = zeros([K,dim,dim]);
         log_weights = log(weights)+(log_normpdf(measurement,selected_model{j}{2}(:,2,1:dim),sqrt(selected_model{j}{3}(:,dim+1:2*dim,dim+1:2*dim))));
         log_weights = log_weights - max(log_weights);
         marginalized_weights = sum(exp(log_weights));
         forward_weights = [forward_weights;marginalized_weights];
         weights = exp(log_weights)./marginalized_weights;
         for i=1:K
            mus = reshape(selected_model{j}{2}(i,:,:),[1,2*dim]);
            joint_cov = reshape(selected_model{j}{3}(i,:,:),[dim*2,dim*2]);
            precision = inv(joint_cov);
            pos_cov = inv(precision(1:dim,1:dim));
            conditional_pre = precision(1:dim,dim+1:2*dim);
            mu_pos = mus(1,1:dim)-pos_cov*conditional_pre*(measurement-mus(1,dim+1:2*dim));
            means(i,:,:) = mu_pos;
            covs(i,:,:) = pos_cov;
         end
         [posterior_mean,posterior_cov]=compute_moments_of_gmm({weights,means,covs});
         forward_means(j,:,:) = posterior_mean;
         forward_covs(j,:,:) = posterior_cov;
    end
    forward_weights = forward_weights./sum(forward_weights);
    forward_message = {forward_weights,forward_means,forward_covs};
end

function [current_max_d,sample_cates]=particle_filter_decision_sssm(samples_of_last,sssm,measurement_model,K,D,num_of_samples,fixed_d)
    current_max_d = 0;
    current_max = 0;
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    rho = sssm{1};
    sample_weights = samples_of_last{1};
    sample_categories = samples_of_last{2};
    sample_means = samples_of_last{3};
    sample_vars = samples_of_last{4};
    sample_states = samples_of_last{5};
    sample_cates = sample_categories;
    for d = 1:D
        if fixed_d ==0 ||fixed_d ==d
            new_sample_cates = zeros([num_of_samples,1]);
            predictive_means = zeros([num_of_samples,1]);
            predictive_vars = zeros([num_of_samples,1]);
            new_sample_weights = zeros([num_of_samples,1]);
            for s = 1:num_of_samples
                discrete_transition = rho(sample_categories(s),:);
                new_sample_cates(s) = find(mnrnd(1,discrete_transition)==1);
                new_sample_weights(s) = discrete_transition(new_sample_cates(s))*sample_weights(s);
                x_trans = sssm{2}{new_sample_cates(s)}{d};
                coef = x_trans{1};
                mean_val = x_trans{2};
                var_val = x_trans{3};
                predictive_vars(s) = var_val+coef*sample_vars(s)*coef';
                predictive_means(s) = coef*sample_states(s)+mean_val;
            end
            new_sample_states = normrnd(predictive_means,sqrt(predictive_vars));
            new_sample_measurments = normrnd(H*new_sample_states+b,sqrt(R));
            
            state_marginal_probs = zeros([1,num_of_samples]);
            measurment_marginal_probs = zeros([1,num_of_samples]);
            joint_probs = zeros([1,num_of_samples]);
            for s=1:num_of_samples
                state_sampled = new_sample_states(s);
                measurement_sampled = new_sample_measurments(s);
                indices = setdiff(1:num_of_samples,s);
                samples_excludes_s=new_sample_states(indices);
                sample_weights_excludes_s=new_sample_weights(indices);
    
    
                predictive_state_probs = zeros([K,num_of_samples-1]);
                measurement_prob = zeros([K,num_of_samples-1]);
                for n = 1:K
                    x_trans = sssm{2}{n}{d};
                    coef = x_trans{1};
                    mean_val = x_trans{2};
                    var_val = x_trans{3};
                    predictive_means = coef*samples_excludes_s+mean_val;
                    predictive_state_probs(n,:) = normpdf(state_sampled,predictive_means,sqrt(var_val));
                    measurement_prob(n,:)=normpdf(measurement_sampled,H*predictive_means+b,sqrt(H*var_val*H'+R));
                end
                weights = rho(sample_categories(s),:)';
                predictive_state_probs = sum(predictive_state_probs.*weights,1);
                state_marginal_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs);
                joint_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs.*normpdf(measurement_sampled,H*state_sampled+b,sqrt(R)));
                measurement_prob = sum(measurement_prob.*weights,1);
                measurment_marginal_probs(s) = dot(sample_weights_excludes_s,measurement_prob);
            end
            state_marginal_probs(state_marginal_probs==0)=1;
            measurment_marginal_probs(measurment_marginal_probs==0)=1;
            joint_probs(joint_probs==0)=1;
            predictive_state_entropy = -dot(sample_weights,log(state_marginal_probs));
            predictive_measuerment_entropy = -dot(sample_weights,log(measurment_marginal_probs));
            predictive_joint_entropy = -dot(sample_weights,log(joint_probs));
            mi = predictive_state_entropy+predictive_measuerment_entropy-predictive_joint_entropy;
            if mi>=current_max
                current_max = mi;
                current_max_d =d;
                sample_cates = new_sample_cates;
            end
        end
        
    end

end

function [samples]=reweight_particles(num_of_samples,new_sample_cates,max_d,samples_of_last,sssm,measurement_model,measurement,K,effective_sample_threshold)
    sample_weights = samples_of_last{1};
    sample_cates = samples_of_last{2};
    sample_means = samples_of_last{3};
    sample_vars = samples_of_last{4};
    last_sample_states = samples_of_last{5};
    rho = sssm{1};    
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    weights_new = zeros([1,num_of_samples]);
    means_new = zeros([1,num_of_samples]);
    vars_new = zeros([1,num_of_samples]);
    for s=1:num_of_samples
        discrete_transition = rho(sample_cates(s),:);
        x_trans = sssm{2}{new_sample_cates(s)}{max_d};
        sample_weights(s) = discrete_transition(new_sample_cates(s))*sample_weights(s);
        coef = x_trans{1};
        mean_val = x_trans{2};
        var_val = x_trans{3};
        mu_bar = coef*sample_means(s)+mean_val;
        sigma_bar = var_val+coef*sample_vars(s)*coef';
        var_new = inv(inv(sigma_bar)+H'/R*H);
        vars_new(s) = var_new;
        means_new(s) = var_new*(H'/R*(measurement-b)+sigma_bar\mu_bar);
        weights_new(s) = log_normpdf(measurement,H*(mu_bar)+b,sqrt(R+H*sigma_bar*H'));
    end
    new_sample_states = normrnd(means_new,sqrt(vars_new));
    log_weights_new = log(sample_weights)+weights_new;
    log_weights_new = log_weights_new-max(log_weights_new);
    weights_new = exp(log_weights_new)./sum(exp(log_weights_new));
    samples = {weights_new,new_sample_cates,means_new,vars_new,new_sample_states};
    sample_efficiency = 1/sum(weights_new.^2);
    if sample_efficiency<effective_sample_threshold
         sample_copies = mnrnd(num_of_samples,weights_new);
         new_sample_cates = repelem(new_sample_cates,sample_copies);
         means_new = repelem(means_new,sample_copies);
         vars_new = repelem(vars_new,sample_copies);
         weights_new = ones([1,num_of_samples]).*(1/num_of_samples);
         samples = {weights_new,new_sample_cates,means_new,vars_new,new_sample_states};
    end
   
end

function MI=compute_augmented_dist_MI_MC(approximated_augmented_dist,sample_size)
    weights = approximated_augmented_dist{1};
    means = approximated_augmented_dist{2};
    covs = approximated_augmented_dist{3};
    sampled_times = mnrnd(sample_size,weights);
    mean_X = squeeze(means(:,1));
    mean_Y = squeeze(means(:,2));
    var_X = squeeze(covs(:,1,1));
    var_Y = squeeze(covs(:,2,2));
    MI = 0;
    for i = 1:length(sampled_times)
        sampled_xy=mvnrnd(means(i,:),squeeze(covs(i,:,:)),sampled_times(i));
        sampled_x = squeeze(sampled_xy(:,1));
        sampled_y = squeeze(sampled_xy(:,2));
        prob_xy = zeros([sampled_times(i),length(weights)]);
        prob_x = zeros([sampled_times(i),length(weights)]);
        prob_y = zeros([sampled_times(i),length(weights)]);
        for k=1:length(weights)
            prob_xy(:,k)=mvnpdf(sampled_xy,means(k,:),squeeze(covs(k,:,:)));
            prob_x(:,k)=normpdf(sampled_x,mean_X(k),sqrt(var_X(k)));
            prob_y(:,k)=normpdf(sampled_y,mean_Y(k),sqrt(var_Y(k)));
        end
        MI = MI + sum(log(prob_xy*weights)-log(prob_x*weights)-log(prob_y*weights));
    end
    MI = MI/sample_size;
end