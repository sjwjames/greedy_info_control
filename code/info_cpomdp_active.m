function [ADF_results,PF_results]=info_cpomdp_active(agent_transition_models,target_transition_model,initial_belief,initial_target_state,initial_agent_state,N,T,state_boundary,preset_path,verbose,run_flags)
%     rng(1);
    n_of_measurement_samples = 1;
    distance_threshold = 1;

    rewards_chosen = [];
    rewards_collections = [];
    agent_states = [];
    target_states=[];
    measurements=[];
    ADF_decision_results = [];
    ADF_beliefs = {};
    ADF_stop_flag = ~run_flags(1);
    ADF_results = {};
    ADF_time = [];

    PF_stop_flag = ~run_flags(2);
    PF_results = {};
    PF_agent_states = [];
    PF_target_states = [];
    PF_measurements=[];
    PF_decision_results = [];
    PF_beliefs = {};
    PF_n_samples = 1000;
    PF_time = [];
    initial_weights = initial_belief{1};
    initial_means = initial_belief{2};
    initial_vars = initial_belief{3};
    assignments = mnrnd(PF_n_samples,initial_weights);
    initial_state_samples = [];
    for i=1:length(initial_weights)
        samples = normrnd(initial_means(i),sqrt(initial_vars(i)),[assignments(i),1]);
        initial_state_samples = [initial_state_samples;samples];
    end
    PF_initial_belief = {ones([PF_n_samples,1]).*(1/PF_n_samples),initial_state_samples};

   

    for t=1:T
        if verbose
            disp("T="+string(t))
        end
        %         ADF procedure
        ADF_last_belief = initial_belief;
        ADF_last_x = initial_agent_state;
        if t~=1
            ADF_last_belief = ADF_beliefs{t-1};
            ADF_last_x = agent_states(t-1);
        end
        preset_decision = 0;
        if ~isempty(preset_path)
            preset_decision = preset_path{1}(t);
        end
        ADF_start = tic;
        [current_max_d,pred_state_marg,z_samples,current_max_MI_reward]=cpomdp_active_make_decision(ADF_last_belief,agent_transition_models,target_transition_model,preset_decision,N,ADF_last_x);
        ADF_end = toc(ADF_start);
        ADF_time = [ADF_time,ADF_end];
        ADF_decision_results = [ADF_decision_results;current_max_d];

        % simulate to next step, states hidden to the agent
        current_target_state = initial_target_state;
        if t~=1
            current_target_state = target_states(t-1);
        end
        next_target_state = target_transition(current_target_state,target_transition_model);
        target_states = [target_states;next_target_state];
        PF_target_states = [PF_target_states;next_target_state];
        
        next_agent_state = simulate_agent(current_max_d,agent_transition_models,ADF_last_x);
        agent_states = [agent_states;next_agent_state];

        next_measurement = target_measurement(next_target_state,next_agent_state);
        measurements = [measurements;next_measurement];

        %         update belief
        ADF_beliefs{t} = update_posterior(pred_state_marg,next_measurement,next_agent_state,z_samples);


        % if abs(next_measurement) < distance_threshold || t==T
        %     ADF_stop_flag = true;
        %
        % end


        PF_current_belief = PF_initial_belief;
        PF_current_agent_state = initial_agent_state;
        if t~=1
            PF_current_belief = PF_beliefs{t-1};
            PF_current_agent_state = PF_agent_states(t-1);
        end
        pf_start = tic;
        [PF_decision,PF_sampled_target_states]=PF_cpomdp_dm_update(PF_current_belief,target_transition_model,agent_transition_models,PF_current_agent_state);
        pf_time_spent = toc(pf_start);
        PF_time = [PF_time,pf_time_spent];
        PF_decision_results = [PF_decision_results;PF_decision];
        PF_next_agent_state = simulate_agent(PF_decision,agent_transition_models,PF_current_agent_state);
        PF_agent_states = [PF_agent_states;PF_next_agent_state];
        PF_next_measurement = target_measurement(next_target_state,PF_next_agent_state);
        PF_measurements = [PF_measurements;PF_next_measurement];
        

        PF_posterior_belief = pf_update_belief(PF_next_measurement,target_transition_model,PF_sampled_target_states,PF_current_belief,PF_next_agent_state);
        PF_beliefs{t} = PF_posterior_belief;
        
        
        
        %              if PF_decision==3 || t==T
        %  if t==T || rewards_chosen_PF(3)>0
        %     PF_stop_flag = true;
        %
        % end


        % if ADF_stop_flag&&PF_stop_flag
        %     return;
        % end
    end
    ADF_results = {initial_target_state,ADF_decision_results,agent_states,target_states,measurements,ADF_beliefs,ADF_time};
    PF_results = {initial_target_state,PF_decision_results,PF_agent_states,PF_target_states,PF_measurements,PF_beliefs,PF_time};
end



function [current_max_d,pred_state_marg,z_samples,current_max_MI_reward]=cpomdp_active_make_decision(last_belief,agent_transition_models,target_transition_model,preset_decision,N,last_x)
    n_of_actions = length(agent_transition_models);
    
    current_max_d = 0;
    current_max_MI_reward = -inf;
    target_trans_weights = target_transition_model{1};
    target_trans_coefs = target_transition_model{2};
    target_trans_means = target_transition_model{3};
    target_trans_vars = target_transition_model{4};
    pred_state_marg_ws = [];
    pred_state_marg_ms = [];
    pred_state_marg_vs = [];
    prior_ws = last_belief{1};
    prior_ms = last_belief{2};
    prior_vs = last_belief{3};
    for n=1:length(target_trans_weights)
        pred_state_marg_ws = [pred_state_marg_ws;prior_ws.*target_trans_weights(n)];
        pred_state_marg_ms = [pred_state_marg_ms;prior_ms.*target_trans_coefs(n)+target_trans_means(n)];
        pred_state_marg_vs = [pred_state_marg_vs;target_trans_vars(n)+target_trans_coefs(n)*prior_vs*target_trans_coefs(n)'];
    end
    pred_state_marg = {pred_state_marg_ws,pred_state_marg_ms,pred_state_marg_vs};
    z_samples = sample_from_gmm(pred_state_marg,N);

    for d=1:n_of_actions
        if preset_decision==0 || preset_decision==d
            transition_model = agent_transition_models{d};
            pred_agent_state_dist = {transition_model{1},transition_model{2}.*last_x+transition_model{3},transition_model{4}};
            agent_state_samples = sample_from_gmm(pred_agent_state_dist,N);
            mis = [];
            marg_entropies = [];
            for i=1:N
                agent_state = agent_state_samples(i);
                mms = get_measurement_moments(z_samples,agent_state);
                marg_ll = {ones([N,1])/N,mms{1},mms{2}};
                [marg_mean,marg_val] = compute_moments_of_gmm(marg_ll);
                mis = [mis,0.5*log(marg_val)-mean(0.5*log(mms{2}))];

            end

            % for i = 1:N
            %     target_state = z_samples(i);
            %     mms = get_measurement_moments(target_state,agent_state_samples);
            %     marg_ll = {ones([N,1])/N,mms{1},mms{2}};
            %     [marg_mean,marg_val] = compute_moments_of_gmm(marg_ll);
            %     marg_entropies = [marg_entropies;0.5*log(marg_val)+0.5+0.5*log(2*pi)];
            % end
            % cond_entropies = [];
            % for i=1:N
            %     agent_state = agent_state_samples(i);
            %     mms = get_measurement_moments(z_samples,agent_state);
            %     cond_entropies = [cond_entropies;mean(0.5*log(mms{2}))+0.5+0.5*log(2*pi)];
            % end
            mi = mean(mis);
            % mi = mean(marg_entropies)-mean(cond_entropies);
            disp(mi)
            if mi>current_max_MI_reward
                current_max_d = d;
                current_max_MI_reward = mi;
            end
        end
        
    end

    
end


function updated_belief=update_posterior(pred_state_marg,next_measurement,next_agent_state,z_samples)
    N = length(z_samples);
    z_sample_probs = arrayfun(@(x) dot(pred_state_marg{1},normpdf(x,pred_state_marg{2},sqrt(pred_state_marg{3}))),z_samples);
    z_sample_probs = z_sample_probs./sum(z_sample_probs);
    cond_ll_probs=target_measurement_probs(z_samples,next_agent_state,next_measurement);
    cond_ll_probs = cond_ll_probs./sum(cond_ll_probs);
    mms = get_measurement_moments(z_samples,next_agent_state);
    marg_ll = {ones([N,1])/N,mms{1},mms{2}};
    [marg_mean,marg_val] = compute_moments_of_gmm(marg_ll);
    log_marg_ll_prob = log_normpdf(next_measurement,marg_mean,sqrt(marg_val));
    

    log_post_probs = log(z_sample_probs)+log(cond_ll_probs)-log_marg_ll_prob;
    log_post_probs = log_post_probs-max(log_post_probs);
    post_probs = exp(log_post_probs)./sum(exp(log_post_probs));
    mean_val = dot(post_probs,z_samples);
    var_val = dot(post_probs, (z_samples - mean_val).^2);

    updated_belief = {1,mean_val,var_val};
end

function [next_agent_state] = simulate_agent(decision,transition_models,current_agent_state)
    transition_model = transition_models{decision};
    next_agent_state=normrnd(current_agent_state+transition_model{3},sqrt(transition_model{4}));
end

function [next_agent_state,next_measurement,next_target_state]=simulate_process(decision,transition_models,current_agent_state,current_target_state,target_transition_model,preset_path,state_boundary)
    transition_model = transition_models{decision};
    
    if isempty(preset_path) || length(preset_path)<2
        next_agent_state=normrnd(current_agent_state+transition_model{3},sqrt(transition_model{4}));
        
        weights_vector = target_transition_model{1};
        coefficients_vector = target_transition_model{2};
        mu_vector = target_transition_model{3};
        var_vector = target_transition_model{4};

        assignment = mnrnd(1,weights_vector);
        coef = coefficients_vector(assignment==1);
        mu = mu_vector(assignment==1);
        var = var_vector(assignment==1);
        next_target_state = normrnd(coef*current_target_state+mu,sqrt(var));
    else
        next_agent_state = preset_path{2}(t);
        next_target_state = preset_path{4}(t);
    end
    
    if next_agent_state<state_boundary(1)
        next_agent_state=state_boundary(1);
    elseif next_agent_state>state_boundary(2)
        next_agent_state=state_boundary(2);
    end

    if next_target_state<state_boundary(1)
        next_target_state=state_boundary(1);
    elseif next_target_state>state_boundary(2)
        next_target_state=state_boundary(2);
    end

    
    
    if isempty(preset_path) || length(preset_path)<3
        next_measurement = target_measurement(next_target_state,next_agent_state);
    else
        next_measurement = preset_path{3}(t);
    end
   
end

function [current_max_d,sampled_target_states]=PF_cpomdp_dm_update(current_belief,target_transition_model,agent_transition_models,current_agent_state)
    
    % decision-making process
    target_states_of_last = current_belief{2};
    weights_of_last = current_belief{1};
    N = length(target_states_of_last);

    current_max_d = -inf;
    current_max = -inf;

    n_of_actions = length(agent_transition_models);
    
    target_trans_weights = target_transition_model{1};
    target_trans_coefs = target_transition_model{2};
    target_trans_means = target_transition_model{3};
    target_trans_vars = target_transition_model{4};

    pred_state_marg_ws = [];
    pred_state_marg_ms = [];
    pred_state_marg_vs = repelem(target_trans_vars,N);
    for n=1:length(target_trans_weights)
        pred_state_marg_ws = [pred_state_marg_ws;weights_of_last.*target_trans_weights(n)];
        pred_state_marg_ms = [pred_state_marg_ms;target_states_of_last.*target_trans_coefs(n)+target_trans_means(n)];
    end
    pred_state_marg = {pred_state_marg_ws,pred_state_marg_ms,pred_state_marg_vs};
    
    sampled_target_states = sample_from_gmm(pred_state_marg,N);

    rep_weights = repmat(weights_of_last,[N,1]);
    for d=1:n_of_actions
        transition_model = agent_transition_models{d};
        pred_agent_state_dist = {transition_model{1}, transition_model{2} .* current_agent_state + transition_model{3}, transition_model{4}};
        agent_state_samples = sample_from_gmm(pred_agent_state_dist, N);
        
        
        mis = zeros(1, N);
        rep_target_states = repmat(sampled_target_states,[N,1]); 
        rep_agent_states = repelem(agent_state_samples,N);
        rep_measurements = target_measurement(rep_target_states, rep_agent_states);
        
        for i = 1:N
            agent_state = agent_state_samples(i);
            meaurements_sampled = rep_measurements((i-1)*N+1:i*N);
            marg_ll = target_measurement_probs(rep_target_states, agent_state, repelem(meaurements_sampled,N));
            marg_ll = mean(reshape(marg_ll,[N,N]),1);
            % marg_ll = marg_ll./sum(marg_ll);
            nonzero_marg_ll = marg_ll(marg_ll~=0);
            mms = get_measurement_moments(sampled_target_states, agent_state);
            mis(i) = -mean(log(nonzero_marg_ll)) - mean(0.5 * log(mms{2}))-0.5*log(2*pi)-0.5;
        end

         % rep_measurements_total = [];
        % rep_agent_total = repelem(agent_state_samples,N*N);
        % rep_target_total = repmat(rep_target_states,[N,1]);
        % for i=1:N
        %     meaurements_sampled = rep_measurements((i-1)*N+1:i*N);
        %     rep_measurements_total = [rep_measurements_total;repelem(meaurements_sampled,N)];
        % end
        % 
        % marg_ll_total = target_measurement_probs(rep_target_states, agent_state, repelem(meaurements_sampled,N));
        % for i = 1:N
        %     agent_state = agent_state_samples(i);
        % 
        %     marg_ll = marg_ll_total((i-1)*N*N+1:i*N*N);
        %     % marg_ll = mean(reshape(marg_ll,[N,N]),1);
        %     mms = get_measurement_moments(sampled_target_states, agent_state);
        %     mis(i) = -mean(log(marg_ll),"all") - mean(0.5 * log(mms{2}))-0.5*log(2*pi)-0.5;
        % end

        
        mi = mean(mis);
        disp(mi);
        if mi>=current_max
            current_max = mi;
            current_max_d =d;
            
        end
    end
    
    
end

function posterior_belief = pf_update_belief(next_measurement,target_transition_model,sampled_target_states,current_belief,next_agent_state)
    target_states_of_last = current_belief{2};
    weights_of_last = current_belief{1};
    N = length(weights_of_last);
    effective_sample_threshold = N/2;
    % update belief
    weights = target_transition_model{1};
    coefficients = target_transition_model{2};
    means = target_transition_model{3};
    variance = target_transition_model{4};
    
    z_marg_probs = [];
    for i=1:length(weights)
        probs = weights(i)*normpdf(sampled_target_states,coefficients(i).*target_states_of_last+means(i),sqrt(variance(i)));
        z_marg_probs = [z_marg_probs;probs];
    end
    z_sample_probs = sum(z_marg_probs,1);
    cond_ll_probs=target_measurement_probs(sampled_target_states,next_agent_state,next_measurement);
    log_weights_new = log(cond_ll_probs)+log(z_sample_probs)+log(weights_of_last);
    log_weights_new = log_weights_new-max(log_weights_new);
    weights_new = exp(log_weights_new)./sum(exp(log_weights_new));

    samples_new = sampled_target_states;
    
    sample_efficiency = 1/sum(weights_new.^2);
    if sample_efficiency<effective_sample_threshold
         sample_copies = mnrnd(N,weights_new);
         samples_new = repelem(samples_new,sample_copies);
         weights_new = ones([N,1]).*(1/N);
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
        if assignments(i)>0
            s = normrnd(means(i),sqrt(vars(i)),[assignments(i),1]);
            samples = [samples;s];
        end
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


function next_target_states = target_transition(current_target_states,target_transition_model)
    weights_vector = target_transition_model{1};
    coefficients_vector = target_transition_model{2};
    mu_vector = target_transition_model{3};
    var_vector = target_transition_model{4};
        
    assignment = mnrnd(1,weights_vector);
    coef = coefficients_vector(assignment==1);
    mu = mu_vector(assignment==1);
    var = var_vector(assignment==1);
    next_target_states = normrnd(coef*current_target_states+mu,sqrt(var));
end

function results=target_measurement(z,x)
    moments = get_measurement_moments(z,x);
    results = normrnd(moments{1},sqrt(moments{2})); 
end

function probs=target_measurement_probs(z,x,y)
    moments = get_measurement_moments(z,x);
    probs = normpdf(y,moments{1},sqrt(moments{2})); 
end


