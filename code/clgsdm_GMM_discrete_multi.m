% Closed-loop greedy information control in high dimensions with
% multivirate states and measurements. In this process, decisions are made
% online and their culmulative MI is computed along the process.
% Considering the computational issue, discretization method is temporarily
% adopted and the PF results work as a surrogate ground truth
function [EP_results,ADF_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results]= clgsdm_GMM_discrete_multi(initial_state,initial_model,transition_models,measurement_model,T,K,state_dim,measurement_dim,convergence_threshold,fixed_settings,rnd_seed)
    debug_flag = true;
    EP_decision_results = [];
    ADF_decision_results = [];
    ADF_simple_decision_results = [];
    ADF_gmm_unc_decisions = [];
    discretized_decisions = [];
    pf_decision_results = [];

    % prior distribution p_0
    initial_mean = initial_model{1};
    initial_cov = initial_model{2};
    
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};

    num_of_samples = 1000;
    sample_weights = ones([1,num_of_samples]).*(1/num_of_samples);
    samples_of_states = mvnrnd(initial_mean,initial_cov,num_of_samples);
    samples_of_measurements = mvnrnd(samples_of_states*H'+b,R);
    samples_of_last = [samples_of_states,samples_of_measurements];
    effective_sample_threshold =100;

    EP_states=zeros(T,state_dim);
    EP_measurements = zeros(T,measurement_dim);
    ADF_states=zeros(T,state_dim);
    ADF_measurements = zeros(T,measurement_dim);
    ADF_gmm_unc_states = zeros(T,state_dim);
    ADF_gmm_unc_measurements = zeros(T,measurement_dim);
    ADF_simple_states=zeros(T,state_dim);
    ADF_simple_measurements = zeros(T,measurement_dim);
    pf_states = zeros(T,state_dim);
    pf_measurements = zeros(T,measurement_dim);
    
    
    num_of_comp = length(transition_models{1}{1});

    EP_forward_messages = {{zeros([1,state_dim]),zeros([state_dim,state_dim])}};
    EP_backward_messages = {{zeros([1,state_dim]),zeros([state_dim,state_dim])}};
    ADF_forward_messages = {{ones([num_of_comp,1]).*(1/num_of_comp),ones([state_dim,num_of_comp]).*initial_mean,ones([state_dim,state_dim,num_of_comp]).*initial_cov}};
    ADF_gmm_unc_inference = {{ones([num_of_comp,1]).*(1/num_of_comp),ones([state_dim,num_of_comp]).*initial_mean,ones([state_dim,state_dim,num_of_comp]).*initial_cov}};
    ADF_simple_forward_messages = {{zeros([1,state_dim]),zeros([state_dim,state_dim])}};

    random_decisions = randi([1,K],1,T+5);
    random_states = zeros(T,state_dim);
    random_measurements = zeros(T,measurement_dim);

    fixed_states_measurements = {};
    fixed_decisions = {};
    fixed_states = {};
    fixed_measurements = {};
    if ~isempty(fixed_settings)
        fixed_states_measurements = fixed_settings{1};
        fixed_decisions = fixed_settings{2};
        fixed_states = fixed_states_measurements{1};
        fixed_measurements = fixed_states_measurements{2};
    end
    
    for i=1:T
        last_state = initial_state;
        if i~=1
            last_state = random_states(i-1,:);
        end
        next_state = dynamic_transition_multi(last_state,random_decisions(i),transition_models,rnd_seed);
        measurement = measure_multi(next_state,measurement_model,rnd_seed);
        if ~isempty(fixed_states_measurements)
            next_state = fixed_states(i,:);
            measurement = fixed_measurements(i,:);
        end
        
        random_states(i,:)=next_state';
        
        random_measurements(i,:)=measurement;   
    end
    
    random_results = {random_decisions,random_states,random_measurements};
    
    fid = fopen("experiment_results/exp.csv",'w');
    fprintf(fid,'t,iterations until convergence,forward steps skipped,backward steps skipped\n');
    fclose(fid);

    ADF_time = zeros([1,T]);
    ADF_gmm_unc_time = zeros([1,T]);
    ADF_simple_time = zeros([1,T]);
    EP_time = zeros([1,T]);
    pf_time = zeros([1,T]);
    pf_mean = zeros([T,state_dim]);
    PF_variance = zeros([T,state_dim]);
    EP_results = {};
    ADF_results = {};
    ADF_gmm_unc_results = {};
    ADF_simple_results = {};
    pf_results = {};
    
    for t=1:T
        EP_forward_messages{t} = {zeros([1,state_dim]),zeros([state_dim,state_dim])};
        EP_backward_messages{t} = {zeros([1,state_dim]),zeros([state_dim,state_dim])};
        
        EP_last_mean = initial_mean;
        EP_last_cov = initial_cov;

        ADF_simple_last_mean = initial_mean;
        ADF_simple_last_cov = initial_cov;

        if t~=1
           EP_last_forward_message = EP_forward_messages{t-1};
           EP_last_cov = inv(EP_last_forward_message{2});
           EP_last_mean = (EP_last_forward_message{2}\EP_last_forward_message{1}')';
            
           ADF_simpe_last_forward_message = ADF_simple_forward_messages{t-1};
           ADF_simple_last_cov = inv(ADF_simpe_last_forward_message{2});
           ADF_simple_last_mean = (ADF_simpe_last_forward_message{2}\ADF_simpe_last_forward_message{1}')';
            
        end

        EP_start = tic;
        if ~isempty(fixed_decisions)
            current_max_d = fixed_decisions(t);
        else
            current_max_d = make_decision_multi(EP_last_mean,EP_last_cov,transition_models,measurement_model,K,state_dim,measurement_dim);
            
        end
        EP_telapsed = toc(EP_start);
        EP_decision_results=[EP_decision_results,current_max_d];

%         ADF_last_message = ADF_forward_messages{t};
%         ADF_start = tic;
%         [ADF_current_max_d,ADF_m_s,ADF_v_s,ADF_y_M,ADF_y_V,ADF_MI_estimations,constrained_true_MI,ADF_gmm_pred_dists]=make_decision_constrained(ADF_last_message,transition_models,measurement_model,K,discretized_states,discretized_measurements);
%         ADF_telapsed = toc(ADF_start);
%         ADF_time(t) = ADF_telapsed;
%         ADF_decision_results=[ADF_decision_results,ADF_current_max_d];
%          
%         ADF_gmm_unc_last_message = ADF_gmm_unc_inference{t};
%         ADF_gmm_unc_start = tic;
%         [ADF_gmm_unc_current_max_d,ADF_gmm_unc_m_s,ADF_gmm_unc_v_s,ADF_gmm_unc_y_M,ADF_gmm_unc_y_V,ADF_gmm_unc_MI_estimations,gmm_unc_constrained_true_MI,ADF_gmm_unc_pred_dists]=make_decision_constrained(ADF_gmm_unc_last_message,transition_models,measurement_model,K,discretized_states,discretized_measurements);
%         ADF_gmm_unc_elapsed = toc(ADF_gmm_unc_start);
%         ADF_gmm_unc_time(t) = ADF_gmm_unc_elapsed;
%         ADF_gmm_unc_decisions=[ADF_gmm_unc_decisions,ADF_gmm_unc_current_max_d];
         
        ADF_simple_start = tic;
        [ADF_simple_current_max_d,ADF_simple_MI_estimations,ADF_simple_pred_dists] = make_decision_multi(ADF_simple_last_mean,ADF_simple_last_cov,transition_models,measurement_model,K,state_dim,measurement_dim);
        ADF_simple_telapsed = toc(ADF_simple_start);
        ADF_simple_time(t) = ADF_simple_telapsed;
        ADF_simple_decision_results=[ADF_simple_decision_results,ADF_simple_current_max_d];
        
         
        pf_start = tic;
        pf_current_max_d = particle_filter_decision_multi(num_of_samples,num_of_comp,K,sample_weights,samples_of_last,transition_models,measurement_model,state_dim,measurement_dim,rnd_seed);
        pf_telapsed = toc(pf_start); 
        pf_decision_results=[pf_decision_results,pf_current_max_d];
        pf_time(t) = pf_telapsed;
        
         EP_last_state = initial_state;
         
         ADF_simple_last_state = initial_state;
%          ADF_last_state = initial_state;
%          ADF_gmm_unc_last_state = initial_state;
         pf_last_state = initial_state;
         if t~=1
            EP_last_state = EP_states(t-1,:);
%             ADF_last_state = ADF_states(t-1,:);
%             ADF_gmm_unc_last_state = ADF_gmm_unc_states(t-1,:);
            ADF_simple_last_state = ADF_simple_states(t-1,:);
            pf_last_state = pf_states(t-1,:);
         end

           % Simulation
        [EP_next_state,EP_next_state_weight] = dynamic_transition_multi(EP_last_state,current_max_d,transition_models,rnd_seed);
        EP_measurement = measure_multi(EP_next_state,measurement_model,rnd_seed);
        if ~isempty(fixed_states_measurements)
            EP_next_state = fixed_states(t);
            EP_measurement = fixed_measurements(t);
        end
         EP_states(t,:)=EP_next_state;
         
         EP_measurements(t,:) = EP_measurement;
            
%          [ADF_next_state,ADF_next_state_weight] = dynamic_transition_multi(ADF_last_state,ADF_current_max_d,transition_models,rnd_seed);
%          ADF_measurement = measure_multi(ADF_next_state,measurement_model,rnd_seed);
%          if ~isempty(fixed_states_measurements)
%             ADF_next_state = fixed_states(t);
%             ADF_measurement = fixed_measurements(t);
%          end
%          ADF_states(t,:) = ADF_next_state;
%          ADF_measurements(t,:) =ADF_measurement;
%          
%          [ADF_gmm_unc_next_state,ADF_gmm_unc_next_state_weight] = dynamic_transition_multi(ADF_gmm_unc_last_state,ADF_gmm_unc_current_max_d,transition_models,rnd_seed);
%          ADF_gmm_unc_measurement = measure_multi(ADF_gmm_unc_next_state,measurement_model,rnd_seed);
%          
%          ADF_gmm_unc_states(t,:) = ADF_gmm_unc_next_state;
%          ADF_gmm_unc_measurements(t,:) = ADF_gmm_unc_measurement;
        
      
         [ADF_simple_next_state,ADF_simple_next_state_weight] = dynamic_transition_multi(ADF_simple_last_state,ADF_simple_current_max_d,transition_models,rnd_seed);
         ADF_simple_measurement = measure_multi(ADF_simple_next_state,measurement_model,rnd_seed);
         if ~isempty(fixed_states_measurements)
            ADF_simple_next_state = fixed_states(t);
            ADF_simple_measurement = fixed_measurements(t);
         end
         ADF_simple_states(t,:) = ADF_simple_next_state;
         ADF_simple_measurements(t,:) = ADF_simple_measurement;

         [pf_next_state,pf_next_state_weight] = dynamic_transition_multi(pf_last_state,pf_current_max_d,transition_models,rnd_seed);
         pf_measurement = measure_multi(pf_next_state,measurement_model,rnd_seed);
         if ~isempty(fixed_states_measurements)
            pf_next_state = fixed_states(t);
            pf_measurement = fixed_measurements(t);
         end
         pf_states(t,:) = pf_next_state;
         pf_measurements(t,:) = pf_measurement;

         % end of simulation


         EP_last_mean_cov = initial_model;
         ADF_simple_last_mean_cov = initial_model;
         if t~=1
            EP_last_message = EP_forward_messages{t-1};
            EP_last_mean_cov = {(EP_last_message{2}\EP_last_message{1}')',inv(EP_last_message{2})};
    
            ADF_simple_last_message = ADF_simple_forward_messages{t-1};
            ADF_simple_last_mean_cov = {(ADF_simple_last_message{2}\ADF_simple_last_message{1}')',inv(ADF_simple_last_message{2})};
         end
    
          model = transition_models{current_max_d};
         [EP_forward_mean_cov_new]=update_after_measurement_multi(model,measurement_model,measurement,EP_last_mean_cov,state_dim);
         EP_forward_messages{t}={(EP_forward_mean_cov_new{2}\EP_forward_mean_cov_new{1}')',inv(EP_forward_mean_cov_new{2})};
          
          ADF_simple_model = transition_models{ADF_simple_current_max_d};
         [ADF_simple_forward_mean_cov_new]=update_after_measurement_multi(ADF_simple_model,measurement_model,ADF_simple_measurement,ADF_simple_last_mean_cov,state_dim);
         ADF_simple_forward_messages{t}={(ADF_simple_forward_mean_cov_new{2}\ADF_simple_forward_mean_cov_new{1}')',inv(ADF_simple_forward_mean_cov_new{2})};

         reweight_start = tic;
        pf_model_chosen = transition_models{pf_current_max_d};
        [samples_of_last,sample_weights]=reweight_particles_multi(num_of_samples,num_of_comp,sample_weights,samples_of_last,pf_model_chosen,measurement_model,pf_measurement,effective_sample_threshold,state_dim,measurement_dim);
        reweight_telapsed = toc(reweight_start);
        pf_time(t) = pf_time(t)+reweight_telapsed;
         

          EP_convergence_start = tic;
         [EP_forward_messages,EP_backward_messages,useless_messages,forward_pass_skipped_intotal,backward_pass_skipped_intotal,ite_num]=clgsdm_general_multi(initial_model,transition_models,measurement_model,t,convergence_threshold,EP_measurements,EP_decision_results(1:t),EP_forward_messages,EP_backward_messages,state_dim);
          EP_convergence_telapsed = toc(EP_convergence_start);
         EP_time(t) = EP_telapsed+EP_convergence_telapsed;
           
         if debug_flag
             disp("t="+string(t));
             [EP_means,EP_stds]=compute_mean_std_from_message_multi(EP_forward_messages,EP_backward_messages,t,state_dim);
             [ADF_simple_means,ADF_simple_stds] =compute_mean_std_from_message_multi(ADF_simple_forward_messages,{},t,state_dim);
             disp("EP state inference error:"+string(norm(EP_means(t,:)-EP_states(t,:))));
             disp("ADF state inference error:"+string(norm(ADF_simple_means(t,:)-ADF_states(t,:))))
         end
         pf_mean(t,:)=sample_weights*samples_of_last(:,1:state_dim);
    end
%     ADF_means = zeros([T,dim]);
%     ADF_stds = zeros([dim,dim,T]);
    
    [ADF_simple_means,ADF_simple_stds] =compute_mean_std_from_message_multi(ADF_simple_forward_messages,{},T,state_dim);
%     ADF_gmm_unc_means = zeros([T,dim]);
%     ADF_gmm_unc_stds = zeros([dim,dim,T]);
%     
%     for t=1:T
%         [ADF_mean,ADF_cov] = compute_moments_of_gmm_multi(ADF_forward_messages{t+1});
%         ADF_means (t,:) = ADF_mean;
%         ADF_stds (:,:,t) = ADF_cov;
%         
%         [ADF_gmm_unc_mean,ADF_gmm_unc_cov] = compute_moments_of_gmm_multi(ADF_gmm_unc_inference{t+1});
%         ADF_gmm_unc_means (t,:) = ADF_gmm_unc_mean;
%         ADF_gmm_unc_stds (:,:,t) = ADF_gmm_unc_cov;
%         
%     end
    [EP_means,EP_stds]=compute_mean_std_from_message_multi(EP_forward_messages,EP_backward_messages,T,state_dim);
    
    EP_results = {EP_decision_results,EP_states,EP_measurements,EP_means,EP_stds,EP_time};
%     ADF_results = {ADF_decision_results,ADF_states,ADF_measurements,ADF_means,ADF_stds,ADF_time,ADF_forward_messages,ADF_sampled_components};
%     ADF_gmm_unc_results = {ADF_gmm_unc_decisions,ADF_gmm_unc_states,ADF_gmm_unc_measurements,ADF_gmm_unc_means,ADF_gmm_unc_stds,ADF_gmm_unc_time,ADF_gmm_unc_inference,ADF_gmm_unc_sampled_components};
    ADF_simple_results = {ADF_simple_decision_results,ADF_simple_states,ADF_simple_measurements,ADF_simple_means,ADF_simple_stds,ADF_simple_time};
    pf_results = {pf_decision_results,pf_states,pf_measurements,pf_time,pf_mean};
end


function [current_max_d,MI_estimations,pred_joint_dists]=make_decision_multi(last_mean,last_cov,transition_models,measurement_model,K,state_dim,measurement_dim)
    current_max_d = 0;
    current_max = 0;
    MI_estimations = [];
    pred_joint_dists = {};
    for d=1:K
       model = transition_models{d};
       joint_model=directly_compute_moment_matching_multi(model,measurement_model,last_mean,last_cov,state_dim,measurement_dim);
       [joint_mean,joint_cov]=compute_moments_of_gmm_multi(joint_model,state_dim+measurement_dim);
       pred_joint_dists{d} = {joint_mean,joint_cov};
       likelihood_var = joint_cov(state_dim+1:state_dim+measurement_dim,state_dim+1:state_dim+measurement_dim);
       marginal_var = joint_cov(1:state_dim,1:state_dim);
       current_info_gain = 0.5*log(det(marginal_var))+ 0.5*log(det(likelihood_var))-0.5*log(det(joint_cov));
       MI_estimations = [MI_estimations,current_info_gain];
       if current_info_gain>current_max
          current_max=current_info_gain;
          current_max_d = d;
       end
    end
    
end


function [joint_model] = directly_compute_moment_matching_multi(transition_model,measurement_model,approxmiated_mu,approximated_cov,state_dim,measurement_dim)
    weights_vector = transition_model{1};
    coefficients_vector = transition_model{2};
    mu_vector = transition_model{3};
    covariances = transition_model{4};
    
    num_of_components = length(weights_vector);
    
    
    H=measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    
    
    marginal_components_means=zeros([num_of_components,state_dim]);
    marginal_components_covs=zeros([state_dim,state_dim,num_of_components]);
    for i=1:num_of_components
        marginal_components_means(i,:)=approxmiated_mu*coefficients_vector(:,:,i)+mu_vector(i,:);
        marginal_components_covs(:,:,i)=covariances(:,:,i)+coefficients_vector(:,:,i)*approximated_cov*coefficients_vector(:,:,i)';
    end
    
    joint_dim = state_dim+measurement_dim;
    joint_component_means = zeros([num_of_components,joint_dim]);
    joint_component_covs = zeros([joint_dim,joint_dim,num_of_components]);
    
    for i=1:num_of_components
        joint_component_means(i,:)=[marginal_components_means(i,:),marginal_components_means(i,:)*H+b];
        joint_component_covs(:,:,i)=[marginal_components_covs(:,:,i),marginal_components_covs(:,:,i)*H';H*marginal_components_covs(:,:,i),R+H*marginal_components_covs(:,:,i)*H'];
    end
    
    joint_model = {weights_vector,joint_component_means,joint_component_covs};
end




function [means,stds]=compute_mean_std_from_message_multi(forward_messages,backward_messages,T,dim)
    means = zeros([T,dim]);
    stds = zeros([dim,dim,T]);
    
    
    for t=1:T
        forward_message = forward_messages{t};
        forward_mean = (forward_message{2}\forward_message{1}')';
        forward_std = inv(forward_message{2});
        if isempty(backward_messages)
            means(t,:)=forward_mean;
            stds (:,:,t) = forward_std;
        else
            if t~=T
                backward_message = backward_messages{t};
                forward_backward_mean = ((forward_message{2}+backward_message{2})\(forward_message{1}+backward_message{1})')';
                forward_backward_std = inv((forward_message{2}+backward_message{2}));
                means(t,:)=forward_backward_mean;
                stds (:,:,t) = forward_backward_std;
            else
                means(t,:)=forward_mean;
                stds (:,:,t) = forward_std;
            end
        end
        
    end

end

function [current_max_d]=particle_filter_decision_multi(num_of_samples,num_of_comp,K,sample_weights,samples_of_last,transition_models,measurement_model,state_dim,measurement_dim,rnd_seed)
    current_max_d = 0;
    current_max = 0;
    for d = 1:K
        transition_model = transition_models{d};
        weights = transition_model{1};
        coefficients = transition_model{2};
        means = transition_model{3};
        covs = transition_model{4};
        
        H = measurement_model{1};
        b = measurement_model{2};
        R = measurement_model{3};
        states_of_last = samples_of_last(:,1:state_dim);
        samples = zeros([num_of_samples,state_dim+measurement_dim]);
        
        for s=1:num_of_samples
            samples(s,1:state_dim)= dynamic_transition_multi(states_of_last(s,:),d,transition_models,rnd_seed);
            samples(s,state_dim+1:state_dim+measurement_dim) = mvnrnd(samples(s,1:state_dim)*H'+b,R);
        end
        
        state_marginal_probs = zeros([1,num_of_samples]);
        measurment_marginal_probs = zeros([1,num_of_samples]);
        joint_probs = zeros([1,num_of_samples]);
        for s=1:num_of_samples
%             new state measurement sampled
            state_sampled = samples(s,1:state_dim);
            measurement_sampled = samples(s,state_dim+1:state_dim+measurement_dim);
            
            indices = setdiff(1:num_of_samples,s);
            samples_excludes_s=samples_of_last(indices,1:state_dim);
            sample_weights_excludes_s=sample_weights(indices);
            predictive_means = zeros([num_of_samples-1,state_dim,num_of_comp]);
            measurement_marginal_mean = zeros([num_of_samples-1,measurement_dim,num_of_comp]);
            for n=1:num_of_samples-1
                for k = 1:num_of_comp
                    predictive_means(n,:,k) = samples_excludes_s(n,:)*coefficients(:,:,k)'+means(k,:);
                    measurement_marginal_mean(n,:,k) = predictive_means(n,:,k)*H'+b;
                end
                
            end
            predictive_state_probs = zeros([num_of_comp,num_of_samples-1]);
            measurement_prob = zeros([num_of_comp,num_of_samples-1]);
            for n = 1:num_of_comp
                predictive_state_probs(n,:) = mvnpdf(state_sampled,predictive_means(:,:,n),covs(:,:,n));
                measurement_prob(n,:)=mvnpdf(measurement_sampled,measurement_marginal_mean(:,:,n),H*covs(:,:,n)*H'+R);
            end
            predictive_state_probs = sum(predictive_state_probs.*weights,1);
            state_marginal_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs);
            joint_probs(s) = dot(sample_weights_excludes_s,predictive_state_probs.*mvnpdf(measurement_sampled,state_sampled*H'+b,R));
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
        end
    end

end

function [samples,weights_new]=reweight_particles_multi(num_of_samples,num_of_comp,sample_weights,samples_of_last,transition_model,measurement_model,measurement,effective_sample_threshold,state_dim,measurement_dim)
    weights = transition_model{1};
    coefficients = transition_model{2};
    means = transition_model{3};
    covs = transition_model{4};
        
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    samples = zeros([num_of_samples,state_dim+measurement_dim]);
    weights_new = zeros([1,num_of_samples]);
    
    states_of_last = samples_of_last(:,1:state_dim);
    weight_mean = zeros([num_of_samples,state_dim,num_of_comp]);
    posterior_means = zeros([num_of_samples,state_dim,num_of_comp]);
    comp_covs = zeros([state_dim,state_dim,num_of_comp]);
    for c = 1:num_of_comp
        comp_covs(:,:,c) = inv(inv(covs(:,:,c))+H'/R*H);
    end
    for i =1:num_of_samples
        for c=1:num_of_comp
            weight_mean(i,:,c) = (states_of_last(i,:)*coefficients(:,:,c)'+means(c,:))*H'+b;
            posterior_means(i,:,c) = (weight_mean(i,:,c)/covs(:,:,c)+(measurement-b)*(H'/R)')*comp_covs(:,:,c)';
        end
    end
    weights_posterior = zeros([num_of_samples,num_of_comp]);
    for c=1:num_of_comp
        weight_cov = H*covs(:,:,c)*H'+R;
        weights_posterior(:,c) = mvnpdf(measurement,weight_mean(:,:,c),weight_cov);
    end

    posterior_weights = weights_posterior./sum(weights_posterior,2);
    
    for s=1:num_of_samples
        assignment = mnrnd(1,posterior_weights(s,:));
        mean_val = posterior_means(s,:,assignment==1);
        cov_val = comp_covs(:,:,assignment==1);
        samples(s,1:state_dim) = mvnrnd(mean_val,cov_val);
        samples(s,state_dim+1:state_dim+measurement_dim) = mvnrnd(samples(s,1:state_dim)*H'+b,R);
        weights_new(s) = sample_weights(s)*dot(weights',posterior_weights(s,:));
    end

    weights_new = weights_new./sum(weights_new);
    sample_efficiency = 1/sum(weights_new.^2);
    if sample_efficiency<effective_sample_threshold
         sample_copies = mnrnd(num_of_samples,weights_new);
         samples(:,1) = repelem(samples(:,1),sample_copies);
         samples(:,2) = repelem(samples(:,2),sample_copies);
         weights_new = ones([1,num_of_samples]).*(1/num_of_samples);
    end
   
end

function MI_estimation_multi()
end