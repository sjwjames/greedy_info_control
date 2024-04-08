function [ADF_results,PF_results,ADF_ex_results,PF_ex_results]=info_cpomdp(transition_models,measurement_model,initial_belief,initial_state,M,N,explicit_reward_model,T,state_boundary,preset_path,verbose,run_flags)
%     rng(1);
    n_of_measurement_samples = 10;
    gS = measurement_model{1};
    gO = measurement_model{2};
    K = length(initial_belief);

    rewards_chosen = [];
    rewards_collections = [];
    states=[];
    measurements=[];
    ADF_decision_results = [];
    ADF_beliefs = {};
    alpha = 1;
    ADF_stop_flag = ~run_flags(1);
    ADF_results = {};
    
    ADF_ex_stop_flag = ~run_flags(2);
    ADF_ex_results = {};
    ADF_ex_rewards_chosen = [];
    ADF_ex_rewards_collections = [];
    ADF_ex_states=[];
    ADF_ex_measurements=[];
    ADF_ex_decision_results = [];
    ADF_ex_beliefs = {};

    PF_stop_flag = ~run_flags(3);
    PF_results = {};
    PF_rewards_chosen = [];
    PF_rewards_collections = [];
    PF_states=[];
    PF_measurements=[];
    PF_decision_results = [];
    PF_beliefs = {};
    PF_n_samples = 3000;
    initial_weights = cellfun(@(x)(x{1}),initial_belief);
    initial_means = cellfun(@(x)(x{2}),initial_belief);
    initial_vars = cellfun(@(x)(x{3}),initial_belief);
    assignments = mnrnd(PF_n_samples,initial_weights);
    initial_state_samples = [];
    for i=1:length(initial_weights)
        samples = normrnd(initial_means(i),sqrt(initial_vars(i)),[assignments(i),1]);
        initial_state_samples = [initial_state_samples;samples];
    end
    PF_initial_belief = {ones([PF_n_samples,1]).*(1/PF_n_samples),initial_state_samples};

    PF_ex_stop_flag = ~run_flags(4);
    PF_ex_results = {};
    PF_ex_rewards_chosen = [];
    PF_ex_rewards_collections = [];
    PF_ex_states=[];
    PF_ex_measurements=[];
    PF_ex_decision_results = [];
    PF_ex_beliefs = {};

    rw_stop_flag = true;
    rw_results = {};
    rw_rewards_chosen = [];
    rw_rewards_collections = [];
    rw_states=[];
    rw_measurements=[];
    rw_decision_results = [];
    rw_beliefs = {};

    rw_stop_flag = true;
    rw_ex_results = {};
    rw_ex_rewards_chosen = [];
    rw_ex_rewards_collections = [];
    rw_ex_states=[];
    rw_ex_measurements=[];
    rw_ex_decision_results = [];
    rw_ex_beliefs = {};

    n_rollouts = 1;

    

    for t=1:T
        if verbose
            disp("T="+string(t))
        end
%         ADF procedure
        if ~ADF_stop_flag
            ADF_last_belief = initial_belief;
            if t~=1
                ADF_last_belief = ADF_beliefs{t-1};
            end
            preset_decision = 0;
            if ~isempty(preset_path)
                preset_decision = preset_path{1}(t);
            end
            [current_max_d,max_pred_post,max_phi_ki,max_MI_reward,max_explicit_rewards,rewards] = cpomdp_make_decision(ADF_last_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,false,n_rollouts);
            ADF_decision_results = [ADF_decision_results;current_max_d];
            
            % simulate to next step, states hidden to the agent
            current_state = initial_state;
            if t~=1
                current_state = states(t-1);
            end
            
            transition_model = transition_models{current_max_d};
            if isempty(preset_path) || length(preset_path)<2
                next_state=normrnd(current_state+transition_model{3},sqrt(transition_model{4}));
            else
                next_state = preset_path{2}(t);
            end
            
            if next_state<state_boundary(1)
                next_state=state_boundary(1);
            elseif next_state>state_boundary(2)
                next_state=state_boundary(2);
            end
            max_real_explicit_reward = evaluate_reward(explicit_reward_model{current_max_d},next_state);
            rewards_chosen=[rewards_chosen;max_MI_reward,max_explicit_rewards,max_real_explicit_reward];
            rewards_collections = [rewards_collections;rewards];
            states = [states;next_state];
            if isempty(preset_path) || length(preset_path)<3
                weights_measurement=cellfun(@(x)(normpdf(next_state,x{2},sqrt(x{3}))),gS);
                weights_measurement = weights_measurement./sum(weights_measurement);
                assignment = mnrnd(1,weights_measurement);
                next_measurement = mean(normrnd(gO{assignment==1}{2},sqrt(gO{assignment==1}{3}),[n_of_measurement_samples,1]));
            else
                next_measurement = preset_path{3}(t);
            end
    
            
            measurements = [measurements;next_measurement];
    
    %         update belief
            ADF_beliefs{t} = update_posterior(ADF_last_belief,next_measurement,max_phi_ki,measurement_model,max_pred_post,K);
            
    
            if max_real_explicit_reward > 0 || t==T
                ADF_stop_flag = true;
                ADF_results = {initial_state,ADF_decision_results,states,measurements,ADF_beliefs,rewards_chosen,rewards_collections};
            end
        end
        
        if ~ADF_ex_stop_flag
            ADF_ex_last_belief = initial_belief;
            if t~=1
                ADF_ex_last_belief = ADF_ex_beliefs{t-1};
            end
            preset_decision = 0;
            if ~isempty(preset_path)
                preset_decision = preset_path{1}(t);
            end
            [ADF_ex_current_max_d,ADF_ex_max_pred_post,ADF_ex_max_phi_ki,ADF_ex_max_MI_reward,ADF_ex_max_explicit_rewards,ADF_ex_rewards] = cpomdp_make_decision(ADF_ex_last_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,true,n_rollouts);
            ADF_ex_decision_results = [ADF_ex_decision_results;ADF_ex_current_max_d];
            
            % simulate to next step, states hidden to the agent
            ADF_ex_current_state = initial_state;
            if t~=1
                ADF_ex_current_state = ADF_ex_states(t-1);
            end
            
            transition_model = transition_models{ADF_ex_current_max_d};
            if isempty(preset_path) || length(preset_path)<2
                ADF_ex_next_state=normrnd(ADF_ex_current_state+transition_model{3},sqrt(transition_model{4}));
            else
                ADF_ex_next_state = preset_path{2}(t);
            end
            
            if ADF_ex_next_state<state_boundary(1)
                ADF_ex_next_state=state_boundary(1);
            elseif ADF_ex_next_state>state_boundary(2)
                ADF_ex_next_state=state_boundary(2);
            end
            ADF_ex_max_real_explicit_reward = evaluate_reward(explicit_reward_model{ADF_ex_current_max_d},ADF_ex_next_state);
            ADF_ex_rewards_chosen=[ADF_ex_rewards_chosen;ADF_ex_max_MI_reward,ADF_ex_max_explicit_rewards,ADF_ex_max_real_explicit_reward];
            ADF_ex_rewards_collections = [ADF_ex_rewards_collections;ADF_ex_rewards];
            ADF_ex_states = [ADF_ex_states;ADF_ex_next_state];
            if isempty(preset_path) || length(preset_path)<3
                weights_measurement=cellfun(@(x)(normpdf(ADF_ex_next_state,x{2},sqrt(x{3}))),gS);
                weights_measurement = weights_measurement./sum(weights_measurement);
                assignment = mnrnd(1,weights_measurement);
                ADF_ex_next_measurement = mean(normrnd(gO{assignment==1}{2},sqrt(gO{assignment==1}{3}),[n_of_measurement_samples,1]));
            else
                ADF_ex_next_measurement = preset_path{3}(t);
            end
    
            
            ADF_ex_measurements = [ADF_ex_measurements;ADF_ex_next_measurement];
    
    %         update belief
            ADF_ex_beliefs{t} = update_posterior(ADF_ex_last_belief,ADF_ex_next_measurement,ADF_ex_max_phi_ki,measurement_model,ADF_ex_max_pred_post,K);
            
    
            if ADF_ex_max_real_explicit_reward>0 || t==T
                ADF_ex_stop_flag = true;
                ADF_ex_results = {initial_state,ADF_ex_decision_results,ADF_ex_states,ADF_ex_measurements,ADF_ex_beliefs,ADF_ex_rewards_chosen,ADF_ex_rewards_collections};
            end
        end

        if ~PF_stop_flag
            PF_current_belief = PF_initial_belief;
            PF_current_state = initial_state;
            if t~=1
                PF_current_belief = PF_beliefs{t-1};
                PF_current_state = PF_states(t-1);
            end
            [PF_posterior_belief,PF_next_state,PF_next_measurement,rewards_chosen_PF,PF_rewards,PF_decision]=PF_cpomdp_dm_update(PF_current_belief,transition_models,measurement_model,explicit_reward_model,alpha,preset_decision,false,n_rollouts,state_boundary,PF_current_state,n_of_measurement_samples);
            PF_beliefs{t} = PF_posterior_belief;
            PF_states = [PF_states;PF_next_state];
            PF_measurements = [PF_measurements;PF_next_measurement];
            PF_decision_results = [PF_decision_results;PF_decision];
            PF_rewards_chosen=[PF_rewards_chosen;rewards_chosen_PF];
            PF_rewards_collections = [PF_rewards_collections;PF_rewards];
%              if PF_decision==3 || t==T
             if t==T || rewards_chosen_PF(3)>0
                PF_stop_flag = true;
                PF_results = {initial_state,PF_decision_results,PF_states,PF_measurements,PF_beliefs,PF_rewards_chosen,PF_rewards_collections};
            end
        end

        if ~PF_ex_stop_flag
            PF_ex_current_belief = PF_initial_belief;
            PF_ex_current_state = initial_state;
            if t~=1
                PF_ex_current_belief = PF_ex_beliefs{t-1};
                PF_ex_current_state = PF_ex_states(t-1);
            end
            [PF_ex_posterior_belief,PF_ex_next_state,PF_ex_next_measurement,rewards_chosen_PF_ex,PF_ex_rewards,PF_ex_decision]=PF_cpomdp_dm_update(PF_ex_current_belief,transition_models,measurement_model,explicit_reward_model,alpha,preset_decision,true,n_rollouts,state_boundary,PF_ex_current_state,n_of_measurement_samples);
            PF_ex_beliefs{t} = PF_ex_posterior_belief;
            PF_ex_states = [PF_ex_states;PF_ex_next_state];
            PF_ex_measurements = [PF_ex_measurements;PF_ex_next_measurement];
            PF_ex_decision_results = [PF_ex_decision_results;PF_ex_decision];
            PF_ex_rewards_chosen=[PF_ex_rewards_chosen;rewards_chosen_PF_ex];
            PF_ex_rewards_collections = [PF_ex_rewards_collections;PF_ex_rewards];
%              if PF_decision==3 || t==T
             if t==T || rewards_chosen_PF_ex(3)>0
                PF_ex_stop_flag = true;
                PF_ex_results = {initial_state,PF_ex_decision_results,PF_ex_states,PF_ex_measurements,PF_ex_beliefs,PF_ex_rewards_chosen,PF_ex_rewards_collections};
            end
        end
        
        if ADF_stop_flag&&PF_stop_flag&&ADF_ex_stop_flag&&PF_ex_stop_flag
            return;
        end
    end
end

function [current_max_d,max_pred_post,max_phi_ki,current_max_MI_reward,current_max_explicit_reward,rewards]=cpomdp_make_decision_constrained(last_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,explicit_only_flag,n_rollout)
    n_of_actions = length(transition_models);
    gS = measurement_model{1};
    gO = measurement_model{2};
    current_max_d = 0;
    current_max_MI_reward = -inf;
    current_max_explicit_reward = -inf;
    current_max_reward = -inf;
    max_pred_post = {};
    max_phi_ki = [];
%     the robot is fairly certain about its postion
    prior_weights = cellfun(@(x)(x{1}),last_belief);
    prior_means = cellfun(@(x)(x{2}),last_belief);
    prior_vars = cellfun(@(x)(x{3}),last_belief);
    [approximated_prior_mean,approximated_prior_var] = compute_moments_of_gmm({prior_weights',prior_means',prior_vars'});
    if approximated_prior_var<0.5
        alpha = 0;
    end

    rewards = zeros([1,n_of_actions*3]);
    X = -20:0.1:20;
    Y = 1:0.1:5;
    N_of_sampling = 10;
    for d=1:n_of_actions
        if preset_decision==0 || preset_decision==d
            transition_model = transition_models{d};
            predictive_marginal = last_belief;
            K = length(last_belief);
            pred_joint_weights = zeros([K*M,1]);
            pred_joint_means = zeros([K*M,2]);
            pred_joint_vars = zeros([K*M,2,2]);

            pred_post_dist = cell([K,M]);
            phi_ki_mat = [];

            for k=1:K
                m_k = last_belief{k}{2};
                P_k = last_belief{k}{3};
                w_k = last_belief{k}{1};
                predictive_marginal{k}{2} = m_k+transition_model{3};
                predictive_marginal{k}{3} = P_k+transition_model{4};

                phi_ki = cellfun(@(x)(x{1}*normpdf(x{2},predictive_marginal{k}{2},sqrt(predictive_marginal{k}{3}+x{3}))),gS);
                phi_ki_mat = [phi_ki_mat;phi_ki];
                for m=1:M
                    eta_m = gS{m}{2}/gS{m}{3};
                    Lambda_m = 1/gS{m}{3};
                    eta_mk = eta_m+predictive_marginal{k}{2}/predictive_marginal{k}{3};
                    pre_mk = Lambda_m+1/predictive_marginal{k}{3};

                    mu_i = gO{m}{2};
                    Sigma_i = gO{m}{3};

                    pred_joint_weights(m+(k-1)*K)= w_k*phi_ki(m);
                    pred_joint_means(m+(k-1)*K,:)= [eta_mk/pre_mk,mu_i];
                    pred_joint_vars(m+(k-1)*K,:,:)= [1/pre_mk,0;0,Sigma_i];

                    pred_post_dist{k,m} = {w_k,eta_mk/pre_mk,1/pre_mk};
                end
                
               
            end
            last_weights = cellfun(@(x)(x{1}),last_belief);
            pred_joint_weights = pred_joint_weights./sum(pred_joint_weights,"all");
            [pred_joint_m,pred_joint_v]=compute_moments_of_gmm({pred_joint_weights,pred_joint_means,pred_joint_vars});
            P = pred_joint_v(2,2);
            estimated_mi = 0;
            for j=1:K
                indices = ((j-1)*K+1):(j*K);
                [joint_mean,joint_cov]=compute_moments_of_gmm({pred_joint_weights(indices),pred_joint_means(indices,:),pred_joint_vars(indices,:,:)});
                %             marginal moments of states when s=j
                m_j = joint_mean(1);
                v_j = joint_cov(1,1);
                %             marginal moments of measurements when s=j
                M_tilde_j =  joint_mean(2);
                V_tilde_j = joint_cov(2,2);
                cov_xy = joint_cov(1,2);
                F_j = (cov_xy+m_j*M_tilde_j)/(V_tilde_j+M_tilde_j^2);
                M_j = v_j+m_j^2-(cov_xy+m_j*M_tilde_j)^2/(V_tilde_j+M_tilde_j^2);
                
                
                estimated_mi = estimated_mi+alpha*0.5*sum(pred_joint_weights(indices))*log(det(M_j+F_j*P*F_j')/det(M_j));
            end
            % moment match to a single Gaussian for now
            disp(estimated_mi)
            if explicit_only_flag
                estimated_mi = 0;
            end
            na_MI = MI_na({pred_joint_weights,pred_joint_means,pred_joint_vars},X,Y);
            disp(abs(na_MI-estimated_mi))
            explicit_reward = 0;
    
            
            if ~isempty(explicit_reward_model)
                explicit_reward_func = explicit_reward_model{d};
                arry_pred_joint_weights = reshape(pred_joint_weights,[K*M,1]);
                arry_pred_joint_weights = arry_pred_joint_weights./sum(arry_pred_joint_weights);
                arry_pred_joint_means = reshape(pred_joint_means(:,1),[K*M,1]);
                arry_pred_joint_vars = reshape(pred_joint_vars(:,1,1),[K*M,1]);
                explicit_reward = sum(cellfun(@(x)(x{1}*dot(arry_pred_joint_weights,normpdf(x{2},arry_pred_joint_means,sqrt(arry_pred_joint_vars+x{3})))),explicit_reward_func));
            end
            % immediate reward
            total_reward = estimated_mi+explicit_reward;
            if n_rollout>1
                n = length(pred_joint_weights);
                reshaped_vars = zeros([2,2,n]);
                for i=1:n
                    reshaped_vars(:,:,i) = reshape(pred_joint_vars(i,:,:),[2,2]);
                end
                gm = gmdistribution(pred_joint_means(pred_joint_weights~=0,:),reshaped_vars(:,:,pred_joint_weights~=0),pred_joint_weights(pred_joint_weights~=0));
                Y = random(gm,N_of_sampling);
                future_rewards = 0;
               
                for i=1:N_of_sampling
                    updated_belief=update_posterior(last_belief,Y(i,2),phi_ki_mat,measurement_model,pred_post_dist,K);
                    [max_d,pred_post,phi_ki,max_MI_reward,max_explicit_reward,rewards_next]=cpomdp_make_decision(updated_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,explicit_only_flag,n_rollout-1);
                    future_rewards = future_rewards+max_MI_reward+max_explicit_reward;
                end
                
                total_reward = total_reward+future_rewards/N_of_sampling;
            end

            rewards(d) = estimated_mi;
            rewards(n_of_actions+d) = explicit_reward;
            rewards(2*n_of_actions+d) = total_reward;
%             rewards(3*n_of_actions+d) = na_MI;
            if total_reward>=current_max_reward
                current_max_reward = total_reward;
                current_max_MI_reward = estimated_mi;
                current_max_explicit_reward = explicit_reward;
                current_max_d = d;
                max_pred_post = pred_post_dist;
                max_phi_ki = phi_ki_mat;
            end
        end
        
    end

    
end


function [current_max_d,max_pred_post,max_phi_ki,current_max_MI_reward,current_max_explicit_reward,rewards]=cpomdp_make_decision(last_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,explicit_only_flag,n_rollout)
    n_of_actions = length(transition_models);
    gS = measurement_model{1};
    gO = measurement_model{2};
    current_max_d = 0;
    current_max_MI_reward = -inf;
    current_max_explicit_reward = -inf;
    current_max_reward = -inf;
    max_pred_post = {};
    max_phi_ki = [];
%     the robot is fairly certain about its postion
    prior_weights = cellfun(@(x)(x{1}),last_belief);
    prior_means = cellfun(@(x)(x{2}),last_belief);
    prior_vars = cellfun(@(x)(x{3}),last_belief);
    [approximated_prior_mean,approximated_prior_var] = compute_moments_of_gmm({prior_weights',prior_means',prior_vars'});
    if approximated_prior_var<0.5
        alpha = 0;
    end

    rewards = zeros([1,n_of_actions*3]);
    X = -20:0.1:20;
    Y = 1:0.1:5;
    N_of_sampling = 10;
    for d=1:n_of_actions
        if preset_decision==0 || preset_decision==d
            transition_model = transition_models{d};
            predictive_marginal = last_belief;
            K = length(last_belief);
            pred_joint_weights = zeros([K*M,1]);
            pred_joint_means = zeros([K*M,2]);
            pred_joint_vars = zeros([K*M,2,2]);

            pred_post_dist = cell([K,M]);
            phi_ki_mat = [];

            for k=1:K
                m_k = last_belief{k}{2};
                P_k = last_belief{k}{3};
                w_k = last_belief{k}{1};
                predictive_marginal{k}{2} = m_k+transition_model{3};
                predictive_marginal{k}{3} = P_k+transition_model{4};

                phi_ki = cellfun(@(x)(x{1}*normpdf(x{2},predictive_marginal{k}{2},sqrt(predictive_marginal{k}{3}+x{3}))),gS);
                phi_ki_mat = [phi_ki_mat;phi_ki];
                for m=1:M
                    eta_m = gS{m}{2}/gS{m}{3};
                    Lambda_m = 1/gS{m}{3};
                    eta_mk = eta_m+predictive_marginal{k}{2}/predictive_marginal{k}{3};
                    pre_mk = Lambda_m+1/predictive_marginal{k}{3};

                    mu_i = gO{m}{2};
                    Sigma_i = gO{m}{3};

                    pred_joint_weights(m+(k-1)*K)= w_k*phi_ki(m);
                    pred_joint_means(m+(k-1)*K,:)= [eta_mk/pre_mk,mu_i];
                    pred_joint_vars(m+(k-1)*K,:,:)= [1/pre_mk,0;0,Sigma_i];

                    pred_post_dist{k,m} = {w_k,eta_mk/pre_mk,1/pre_mk};
                end

            end
            prior_weights = cellfun(@(x)(x{1}),last_belief);
            pred_joint_weights = pred_joint_weights./sum(pred_joint_weights,"all");
            % moment match to a single Gaussian for now
            [pred_joint_m,pred_joint_v]=compute_moments_of_gmm({pred_joint_weights,pred_joint_means,pred_joint_vars});
            likelihood_var = pred_joint_v(2,2);
            marginal_var = pred_joint_v(1,1);
            estimated_mi = alpha*(0.5*log(marginal_var)+ 0.5*log(likelihood_var)-0.5*log(det(pred_joint_v)));
            if explicit_only_flag
                estimated_mi = 0;
            end
%             na_MI = MI_na({pred_joint_weights,pred_joint_means,pred_joint_vars},X,Y);
%             disp(na_MI)
            explicit_reward = 0;
    
            
            if ~isempty(explicit_reward_model)
                explicit_reward_func = explicit_reward_model{d};
                arry_pred_joint_weights = reshape(pred_joint_weights,[K*M,1]);
                arry_pred_joint_weights = arry_pred_joint_weights./sum(arry_pred_joint_weights);
                arry_pred_joint_means = reshape(pred_joint_means(:,1),[K*M,1]);
                arry_pred_joint_vars = reshape(pred_joint_vars(:,1,1),[K*M,1]);
                explicit_reward = sum(cellfun(@(x)(x{1}*dot(arry_pred_joint_weights,normpdf(x{2},arry_pred_joint_means,sqrt(arry_pred_joint_vars+x{3})))),explicit_reward_func));
            end
            % immediate reward
            total_reward = estimated_mi+explicit_reward;
            if n_rollout>1
                n = length(pred_joint_weights);
                reshaped_vars = zeros([2,2,n]);
                for i=1:n
                    reshaped_vars(:,:,i) = reshape(pred_joint_vars(i,:,:),[2,2]);
                end
                gm = gmdistribution(pred_joint_means(pred_joint_weights~=0,:),reshaped_vars(:,:,pred_joint_weights~=0),pred_joint_weights(pred_joint_weights~=0));
                Y = random(gm,N_of_sampling);
                future_rewards = 0;
               
                for i=1:N_of_sampling
                    updated_belief=update_posterior(last_belief,Y(i,2),phi_ki_mat,measurement_model,pred_post_dist,K);
                    [max_d,pred_post,phi_ki,max_MI_reward,max_explicit_reward,rewards_next]=cpomdp_make_decision(updated_belief,transition_models,measurement_model,M,explicit_reward_model,alpha,preset_decision,explicit_only_flag,n_rollout-1);
                    future_rewards = future_rewards+max_MI_reward+max_explicit_reward;
                end
                
                total_reward = total_reward+future_rewards/N_of_sampling;
            end

            rewards(d) = estimated_mi;
            rewards(n_of_actions+d) = explicit_reward;
            rewards(2*n_of_actions+d) = total_reward;
%             rewards(3*n_of_actions+d) = na_MI;
            if total_reward>=current_max_reward
                current_max_reward = total_reward;
                current_max_MI_reward = estimated_mi;
                current_max_explicit_reward = explicit_reward;
                current_max_d = d;
                max_pred_post = pred_post_dist;
                max_phi_ki = phi_ki_mat;
            end
        end
        
    end

    
end


function updated_belief=update_posterior(ADF_last_belief,next_measurement,max_phi_ki,measurement_model,max_pred_post,K)
    gO = measurement_model{2};
    log_likelihoods = cellfun(@(x)(log_normpdf(next_measurement,x{2},sqrt(x{3}))),gO);
    log_phi_ki = log(max_phi_ki);
    log_updated_phi_ki = log_phi_ki + log_likelihoods;
    updated_belief = ADF_last_belief;
        
    for k=1:K
        means=cellfun(@(x)(x{2}),max_pred_post(k,:));
        vars=cellfun(@(x)(x{3}),max_pred_post(k,:));
        log_weights = log_updated_phi_ki(k,:);
        log_weights = log_weights-max(log_weights);
        weights = exp(log_weights)./sum(exp(log_weights));
        [mm_mean,mm_var]=compute_moments_of_gmm({weights',means',vars'});
        updated_belief{k}{2} = mm_mean;
        updated_belief{k}{3} = mm_var;
    end
    prior_weights = cellfun(@(x)(x{1}),ADF_last_belief);
    post_weights = exp(log_updated_phi_ki).*prior_weights';
    post_weights = sum(post_weights,2)./sum(post_weights,"all");
    for k=1:K
        updated_belief{k}{1} = post_weights(k);
    end
end

function [next_state,next_measurement]=simulate_process(decision,transition_models,measurement_model,current_state,preset_path,state_boundary,n_of_measurement_samples)
    transition_model = transition_models{decision};
    gS = measurement_model{1};
    gO = measurement_model{2};
    if isempty(preset_path) || length(preset_path)<2
        next_state=normrnd(current_state+transition_model{3},sqrt(transition_model{4}));
    else
        next_state = preset_path{2}(t);
    end
    
    if next_state<state_boundary(1)
        next_state=state_boundary(1);
    elseif next_state>state_boundary(2)
        next_state=state_boundary(2);
    end
    
    if isempty(preset_path) || length(preset_path)<3
        weights_measurement=cellfun(@(x)(normpdf(next_state,x{2},sqrt(x{3}))),gS);
        weights_measurement = weights_measurement./sum(weights_measurement);
        assignment = mnrnd(1,weights_measurement);
        next_measurement = mean(normrnd(gO{assignment==1}{2},sqrt(gO{assignment==1}{3}),[n_of_measurement_samples,1]));
    else
        next_measurement = preset_path{3}(t);
    end
   
end

function [posterior_belief,next_state,next_measurement,rewards_chosen,rewards,current_max_d]=PF_cpomdp_dm_update(current_belief,transition_models,measurement_model,explicit_reward_model,alpha,preset_path,ex_only,n_rollouts,state_boundary,current_state,n_of_measurement_samples)
    
    % decision-making process
    states_of_last = current_belief{2};
    weights_of_last = current_belief{1};
    N = length(states_of_last);
    effective_sample_threshold = N/2;
    gS = measurement_model{1};
    gO = measurement_model{2};
    current_max_d = -inf;
    current_max = -inf;
    max_MI_reward =-inf;
    max_explicit_rewards =-inf;
    n_of_actions = length(transition_models);
    rewards = zeros([1,n_of_actions*3]);
    for d=1:n_of_actions
        sampled_states = dynamic_transition(states_of_last,d,transition_models);
        sampled_measurements = zeros([N,1]);
        weights_measurements=cellfun(@(x)(normpdf(sampled_states,x{2},sqrt(x{3}))),gS,"UniformOutput",false);
        for i=1:N
            weights_measurement = cellfun(@(x)(x(i)),weights_measurements);
            weights_measurement = weights_measurement./sum(weights_measurement);
            assignment = mnrnd(1,weights_measurement);
            sampled_measurements(i) = normrnd(gO{assignment==1}{2},sqrt(gO{assignment==1}{3}));
        end
        transition_model = transition_models{d};
        coefficients = transition_model{2};
        means = transition_model{3};
        variance = transition_model{4};
        
        Ys = cellfun(@(x)(x{1}.*normpdf(sampled_measurements,x{2},sqrt(x{3}))),gO,"UniformOutput",false);        
    
        state_marginal_probs = zeros([N,1]);
        measurment_marginal_probs = zeros([N,1]);
        joint_probs = zeros([N,1]);

        state_marginal_means = states_of_last.*coefficients+means;
        phis = cellfun(@(x)(x{1}.*normpdf(state_marginal_means,x{2},sqrt(x{3}+variance))),gS,"UniformOutput",false);
        cond_means = cellfun(@(x)((x{2}/x{3}+state_marginal_means./variance)./(1/variance+1/x{3})),gS,"UniformOutput",false);   
%         weights_of_last_list = zeros([N*(N-1),1]);
%         state_marginal_means_list = zeros([N*(N-1),1]);
%         sampled_states_list = zeros([N*(N-1),1]);
%         for i=1:N
%             indices = setdiff(1:N,i);
%             indices_to_output = ((i-1)*(N-1)+1):(i*(N-1));
%             weights_of_last_list(indices_to_output) = weights_of_last(indices);
%             state_marginal_means_list(indices_to_output) = state_marginal_means(indices);
%             sampled_states_list(indices_to_output) = sampled_states(indices);
%         end
        
        parfor i =1:N
            indices = setdiff(1:N,i);
            Y = cellfun(@(x)(x(i)),Ys);
            weights_excluded = weights_of_last(indices);
            state_trans_probs = normpdf(sampled_states(i),state_marginal_means(indices),sqrt(variance));
            phi_cavs = cellfun(@(x)(x(indices)),phis,"UniformOutput",false);
            cond_means_cav = cellfun(@(x)(x(indices)),cond_means,"UniformOutput",false);
            measurement_sum_probs = zeros([N-1,1]);
            joint_dist_sum = zeros([N-1,1]);
            for j=1:length(phi_cavs)
                measurement_sum_probs = measurement_sum_probs+phi_cavs{j}.*Y(j);
                joint_dist_sum = joint_dist_sum+phi_cavs{j}.*Y(j).*normpdf(sampled_states(i),cond_means_cav{j},sqrt(1/(1/variance+1/gS{j}{3})));
            end
            measurment_marginal_probs(i) = dot(weights_excluded,measurement_sum_probs);
            state_marginal_probs(i)= dot(weights_excluded,state_trans_probs);
            joint_probs(i) = dot(weights_excluded,joint_dist_sum);
        end
        
        
        mi = 0;
        if ~ex_only&&var(states_of_last,weights_of_last)>0.5
            predictive_state_entropy = -dot(weights_of_last(state_marginal_probs~=0),log(state_marginal_probs(state_marginal_probs~=0)));
            predictive_measuerment_entropy = -dot(weights_of_last(measurment_marginal_probs~=0),log(measurment_marginal_probs(measurment_marginal_probs~=0)));
            predictive_joint_entropy = -dot(weights_of_last(joint_probs~=0),log(joint_probs(joint_probs~=0)));
            mi = alpha*(predictive_state_entropy+predictive_measuerment_entropy-predictive_joint_entropy);
        end
        if mi<0
            disp(1)
        end
        joint_probs = joint_probs./sum(joint_probs);
        explicit_reward_func = explicit_reward_model{d};
        explicit_reward = sum(cellfun(@(x)(x{1}*dot(joint_probs,normpdf(sampled_states,x{2},sqrt(x{3})))),explicit_reward_func));
        reward = mi+explicit_reward;
        rewards(d) = mi;
        rewards(n_of_actions+d) = explicit_reward;
        rewards(2*n_of_actions+d) = reward;
        if reward>=current_max
            current_max = reward;
            current_max_d =d;
            max_MI_reward = mi;
            max_explicit_rewards = explicit_reward;
        end
    end
    % simulation
    [next_state,next_measurement]=simulate_process(current_max_d,transition_models,measurement_model,current_state,preset_path,state_boundary,n_of_measurement_samples);
    max_real_explicit_reward = evaluate_reward(explicit_reward_model{current_max_d},next_state);
    rewards_chosen=[max_MI_reward,max_explicit_rewards,max_real_explicit_reward];
    % update belief
    transition_model = transition_models{current_max_d};
    coefficients = transition_model{2};
    means = transition_model{3};
    variance = transition_model{4};
    state_marginal_means = states_of_last.*coefficients+means;

    samples_new = zeros([N,1]);
    
    marginal_lls = cellfun(@(x)(x{1}.*normpdf(next_measurement,x{2},sqrt(x{3}))),gO);
    phis = cellfun(@(x)(x{1}.*normpdf(state_marginal_means,x{2},sqrt(x{3}+variance))),gS,"UniformOutput",false);
    post_means = cellfun(@(x)((x{2}/x{3}+state_marginal_means./variance)./(1/variance+1/x{3})),gS,"UniformOutput",false);    
    weights_new = zeros([N,1]);
    
    parfor i=1:N
        post_weights = marginal_lls./sum(marginal_lls).*cellfun(@(x)(x(i)),phis);
        weights_new(i) = weights_of_last(i)*sum(post_weights);
        post_weights = post_weights./sum(post_weights);
        assignment = mnrnd(1,post_weights);
        means = cellfun(@(x)(x(i)),post_means);
        samples_new(i)=normrnd(means(assignment==1),sqrt(1/(1/variance+1/gS{assignment==1}{3})));
    end
    

    weights_new = weights_new./sum(weights_new);
    sample_efficiency = 1/sum(weights_new.^2);
    if sample_efficiency<effective_sample_threshold
         sample_copies = mnrnd(N,weights_new);
         samples_new = repelem(samples_new,sample_copies);
         weights_new = ones([1,N]).*(1/N);
    end
    posterior_belief = {weights_new,samples_new};
end



function samples=sample_from_gmm(gmm_model,n_samples)
    weights = gmm_model{1};
    means = gmm_model{2};
    vars = gmm_model{3};
    assignments = mnrnd(n_samples,weights);
    samples=[];
    for i=1:length(assignments)
        s = normrnd(means(i),sqrt(vars(i)),[assignments(i),1]);
        samples = [samples;s];
    end
end

function reward = evaluate_reward(reward_func,state)
    reward = sum(cellfun(@(x)(x{1}.*normpdf(state,x{2},sqrt(x{3}))),reward_func));
end

function MI = MI_na(pred_joint_dist,X,Y)
    delta_state = X(2)-X(1);
    delta_measurement = Y(2)-Y(1);
    [Xv,Yv] = meshgrid(X,Y);
    X_len = length(X);
    Y_len = length(Y);
    P = [Xv(:),Yv(:)];
%     P_cell = num2cell(P,2);
    joint_weights = pred_joint_dist{1};
    joint_means = pred_joint_dist{2};
    joint_vars = pred_joint_dist{3};
    n = length(joint_weights);
    reshaped_vars = zeros([2,2,n]);
    for i=1:n
        reshaped_vars(:,:,i) = reshape(joint_vars(i,:,:),[2,2]);
    end
    gm = gmdistribution(joint_means(joint_weights~=0,:),reshaped_vars(:,:,joint_weights~=0),joint_weights(joint_weights~=0));
%     dists = cellfun(@(xy)(dot(joint_weights,mvnpdf(xy,joint_means(joint_weights~=0,:),reshaped_vars(:,:,joint_weights~=0)))),P_cell);
    dists = pdf(gm,P);
    joint_dist = reshape(dists,[Y_len,X_len]);
    measurement_mariginal = sum(joint_dist,2).*delta_state;
    pred_state_dist = sum(joint_dist,1).*delta_measurement;
    predictive_entropy = -sum(pred_state_dist(pred_state_dist~=0).*log(pred_state_dist(pred_state_dist~=0)).*delta_state);
    measurement_entropy = -sum(measurement_mariginal(measurement_mariginal~=0).*log(measurement_mariginal(measurement_mariginal~=0)).*delta_measurement);
    joint_entropy = -sum(joint_dist(joint_dist~=0).*(log(joint_dist(joint_dist~=0))).*delta_state.*delta_measurement,'all');
    MI = predictive_entropy+measurement_entropy-joint_entropy;
end