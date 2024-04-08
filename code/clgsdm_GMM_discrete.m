function [EP_results,ADF_results,discretized_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results,state_check]=clgsdm_GMM_discrete(initial_state,initial_model,transition_models,measurement_model,T,K,convergence_threshold,num_of_bins,state_range,measurement_range,fixed_settings)
    decision_results=[];
    ADF_decision_results = [];
    ADF_simple_decision_results = [];
    ADF_gmm_unc_decisions = [];
    discretized_decisions = [];
    pf_decision_results = [];
    
    ADF_sampled_components = [];
    EP_sampled_components = [];
    ADF_gmm_unc_sampled_components = [];
    ADF_simple_sampled_components = [];
    discretization_sampled_components = [];
    pf_sampled_components = [];
    
    initial_mean = initial_model{1};
    initial_var = initial_model{2};
    
    states=[];
    measurements=[];
    ADF_states=[];
    ADF_measurements = [];
    ADF_gmm_unc_states = [];
    ADF_gmm_unc_measurements = [];
    ADF_simple_states=[];
    ADF_simple_measurements = [];
    pf_states = [];
    pf_measurements = [];
    
    states_using_discretization=[];
    measurements_using_discretization = [];
    
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
    num_of_comp = length(transition_models{1}{1});
    
    forward_messages = {{0,0}};
    backward_messages = {{0,0}};
    ADF_forward_messages = {{ones([num_of_comp,1]).*(1/num_of_comp),ones([num_of_comp,1]).*initial_mean,ones([num_of_comp,1]).*initial_var}};
    ADF_gmm_unc_inference = {{ones([num_of_comp,1]).*(1/num_of_comp),ones([num_of_comp,1]).*initial_mean,ones([num_of_comp,1]).*initial_var}};
    ADF_simple_forward_messages = {{0,0}};
    step_size = 0.1;
    % generate random decisions, states and measurements
    random_decisions = randi([1,K],1,T+5);
    random_states = [];
    random_measurements = [];
    
    fixed_states_measurements = {};
    fixed_decisions = {};
    fixed_states = {};
    fixed_measurements = {};
    if ~isempty(fixed_settings)
        fixed_states_measurements = fixed_settings{1};
        fixed_decisions = fixed_settings{2};
        fixed_states = fixed_states_measurements(:,1);
        fixed_measurements = fixed_states_measurements(:,2);
    end
    
    
    for i=1:T
        last_state = initial_state;
        if i~=1
            last_state = random_states(i-1);
        end
        next_state = dynamic_transition(last_state,random_decisions(i),transition_models);
        measurement = measure(next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            next_state = fixed_states(i);
            measurement = fixed_measurements(i);
        end
    
        random_states=[random_states;next_state];
    
        random_measurements=[random_measurements;measurement];
    end
    
    random_results = {random_decisions,random_states,random_measurements};
    
    %hard-coded for now
    % state_range = [min(random_states)-6,max(random_states)+6];
    discretized_states = linspace(state_range(1),state_range(2),num_of_bins);
    % measurement_range = [min(random_measurements)-6,max(random_measurements)+6];
    discretized_measurements = linspace(measurement_range(1),measurement_range(2),num_of_bins);
    discretized_state_dist = ones([1,num_of_bins]);
    for i=1:num_of_bins
        discretized_state_dist(i)=normpdf(discretized_states(i),initial_mean,sqrt(initial_var));
    end
    discretized_trans_dist_per_model = {};
    
    measurement_mean_vec = repelem(discretized_states.*H+b,num_of_bins);
    measurement_var_vec = repmat(R,[1,num_of_bins*num_of_bins]);
    rep_discretized_measurements = repmat(discretized_measurements,[1,num_of_bins]);
    measurement_dist = normpdf(rep_discretized_measurements,measurement_mean_vec,sqrt(measurement_var_vec));
    discretized_measurement_dist = reshape(measurement_dist,[num_of_bins,num_of_bins]);
    
    for d=1:K
        model = transition_models{d};
        [state_trans_probs,current_states] = discretize_transition_model(discretized_states,model);
        discretized_trans_dist_per_model{d}=state_trans_probs;
    
    end
    
    num_of_samples = 1000;
    temp_model = transition_models{1};
    num_of_comp = length(temp_model{1});
    sample_weights = ones([1,num_of_samples]).*(1/num_of_samples);
    samples_of_states = normrnd(initial_mean,sqrt(initial_var),[1,num_of_samples]);
    samples_of_measurements = normrnd(samples_of_states.*H+b,sqrt(R),[1,num_of_samples]);
    samples_of_last = [samples_of_states',samples_of_measurements'];
    effective_sample_threshold =100;
    
    fid = fopen("experiment_results/exp.csv",'w');
    fprintf(fid,'t,iterations until convergence,forward steps skipped,backward steps skipped\n');
    fclose(fid);
    
    ADF_time = zeros([1,T]);
    ADF_gmm_unc_time = zeros([1,T]);
    ADF_simple_time = zeros([1,T]);
    EP_time = zeros([1,T]);
    discretize_time = zeros([1,T]);
    pf_time = zeros([1,T]);
    pf_mean = zeros([1,T]);
    pf_variance = zeros([1,T]);
    EP_results = {};
    ADF_results = {};
    ADF_gmm_unc_results = {};
    ADF_simple_results = {};
    discretized_results = {};
    pf_results = {};
    
    for t=1:T
        disp("t="+string(t));
        forward_messages{t} = {0,0};
        backward_messages{t} = {0,0};
        smoothed_means = [];
        smoothed_variances = [];
    
        last_mean = initial_mean;
        last_var = initial_var;
    
        ADF_simple_last_mean = initial_mean;
        ADF_simple_last_var = initial_var;
    
        if t~=1
            last_forward_message = forward_messages{t-1};
            last_var = inv(last_forward_message{2});
            last_mean = last_forward_message{2}\last_forward_message{1};
    
            ADF_simpe_last_forward_message = ADF_simple_forward_messages{t-1};
            ADF_simple_last_var = inv(ADF_simpe_last_forward_message{2});
            ADF_simple_last_mean = ADF_simpe_last_forward_message{2}\ADF_simpe_last_forward_message{1};
    
        end
        
        fixed_decision = 0;
        if ~isempty(fixed_decisions)
            fixed_decision = fixed_decisions(t);
        end
    
        discretize_start = tic;
        [discretized_max_d,current_max_predictive_dist,discretization_MI_estimation,discretize_pred_dists] = make_decision_discretized(discretized_state_dist,discretized_states,discretized_measurements,discretized_trans_dist_per_model,discretized_measurement_dist,num_of_bins,K,fixed_decision);
        discretize_telapsed = toc(discretize_start);
        discretize_time(t) = discretize_telapsed;
        discretized_decisions = [discretized_decisions,discretized_max_d];
    
    
        EP_start = tic;
        current_max_d = make_decision(last_mean,last_var,transition_models,measurement_model,K,fixed_decision);
        EP_telapsed = toc(EP_start);
        decision_results=[decision_results,current_max_d];
    
    
    
        ADF_last_message = ADF_forward_messages{t};
        ADF_start = tic;
        [ADF_current_max_d,ADF_m_s,ADF_v_s,ADF_y_M,ADF_y_V,ADF_MI_estimations,constrained_true_MI,ADF_gmm_pred_dists]=make_decision_constrained(ADF_last_message,transition_models,measurement_model,K,discretized_states,discretized_measurements,fixed_decision);
        ADF_telapsed = toc(ADF_start);
        ADF_time(t) = ADF_telapsed;
        ADF_decision_results=[ADF_decision_results,ADF_current_max_d];
    
        ADF_gmm_unc_last_message = ADF_gmm_unc_inference{t};
        ADF_gmm_unc_start = tic;
        [ADF_gmm_unc_current_max_d,ADF_gmm_unc_m_s,ADF_gmm_unc_v_s,ADF_gmm_unc_y_M,ADF_gmm_unc_y_V,ADF_gmm_unc_MI_estimations,gmm_unc_constrained_true_MI,ADF_gmm_unc_pred_dists]=make_decision_constrained(ADF_gmm_unc_last_message,transition_models,measurement_model,K,discretized_states,discretized_measurements,fixed_decision);
        ADF_gmm_unc_elapsed = toc(ADF_gmm_unc_start);
        ADF_gmm_unc_time(t) = ADF_gmm_unc_elapsed;
        ADF_gmm_unc_decisions=[ADF_gmm_unc_decisions,ADF_gmm_unc_current_max_d];
    
        ADF_simple_start = tic;
        [ADF_simple_current_max_d,ADF_simple_MI_estimations,ADF_simple_pred_dists] = make_decision(ADF_simple_last_mean,ADF_simple_last_var,transition_models,measurement_model,K,fixed_decision);
        ADF_simple_telapsed = toc(ADF_simple_start);
        ADF_simple_time(t) = ADF_simple_telapsed;
        ADF_simple_decision_results=[ADF_simple_decision_results,ADF_simple_current_max_d];
    
    
        pf_start = tic;
        pf_current_max_d = particle_filter_decision(num_of_samples,num_of_comp,K,sample_weights,samples_of_last,transition_models,measurement_model,fixed_decision);
        pf_telapsed = toc(pf_start);
        pf_decision_results=[pf_decision_results,pf_current_max_d];
        pf_time(t) = pf_telapsed;
    
        last_state = initial_state;
        ADF_last_state = initial_state;
        ADF_simple_last_state = initial_state;
        ADF_gmm_unc_last_state = initial_state;
        discretized_last_state = initial_state;
        pf_last_state = initial_state;
        if t~=1
            last_state = states(t-1);
            ADF_last_state = ADF_states(t-1);
            ADF_gmm_unc_last_state = ADF_gmm_unc_states(t-1);
            ADF_simple_last_state = ADF_simple_states(t-1);
            discretized_last_state = states_using_discretization(t-1);
            pf_last_state = pf_states(t-1);
        end
    
        [discretized_next_state,discretized_next_state_weight] = dynamic_transition(discretized_last_state,discretized_max_d,transition_models);
        discretized_measurement = measure(discretized_next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            discretized_next_state = fixed_states(t);
            discretized_measurement = fixed_measurements(t);
        end
        states_using_discretization=[states_using_discretization;discretized_next_state];
        measurements_using_discretization=[measurements_using_discretization;discretized_measurement];
        discretization_sampled_components = [discretization_sampled_components;discretized_next_state_weight];
    
        %          plot MI estimations
        %          X=1:K;
        %          Y = [ADF_simple_MI_estimations;discretization_MI_estimation;ADF_MI_estimations;constrained_true_MI];
        %          bar(X,Y);
        %          hold off
        %          legend('ADF-Gaussian','Discretization','ADF-GMM','Constrained NA','Location','northeast','FontSize', 12)
        %          xlabel('Decisions','FontSize', 14)
        %          ylabel('MI estimation','FontSize', 14)
        %          title("Estimated MI Per Decision at Step "+string(t),'FontSize', 14)
        %          saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/MI_estimation_decision_t='+string(t)+'.pdf');
    
    
        [next_state,EP_next_state_weight] = dynamic_transition(last_state,current_max_d,transition_models);
        measurement = measure(next_state,measurement_model);
        %          next_state = discretized_next_state;
        if ~isempty(fixed_states_measurements)
            next_state = fixed_states(t);
            measurement = fixed_measurements(t);
        end
        states=[states;next_state];
        EP_sampled_components = [EP_sampled_components;EP_next_state_weight];
    
        %          measurement = discretized_measurement;
        measurements=[measurements;measurement];
    
        [ADF_next_state,ADF_next_state_weight] = dynamic_transition(ADF_last_state,ADF_current_max_d,transition_models);
        ADF_measurement = measure(ADF_next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            ADF_next_state = fixed_states(t);
            ADF_measurement = fixed_measurements(t);
        end
        %          ADF_next_state = discretized_next_state;
        ADF_states=[ADF_states;ADF_next_state];
        %          ADF_measurement = discretized_measurement;
        ADF_measurements=[ADF_measurements;ADF_measurement];
        ADF_sampled_components = [ADF_sampled_components;ADF_next_state_weight];
    
        [ADF_gmm_unc_next_state,ADF_gmm_unc_next_state_weight] = dynamic_transition(ADF_gmm_unc_last_state,ADF_gmm_unc_current_max_d,transition_models);
        ADF_gmm_unc_measurement = measure(ADF_gmm_unc_next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            ADF_gmm_unc_next_state = fixed_states(t);
            ADF_gmm_unc_measurement = fixed_measurements(t);
        end
        
        %          ADF_next_state = discretized_next_state;
        ADF_gmm_unc_states=[ADF_gmm_unc_states;ADF_gmm_unc_next_state];
        %          ADF_measurement = discretized_measurement;
        ADF_gmm_unc_measurements=[ADF_gmm_unc_measurements;ADF_gmm_unc_measurement];
        ADF_gmm_unc_sampled_components = [ADF_gmm_unc_sampled_components;ADF_gmm_unc_next_state_weight];
    
    
        [ADF_simple_next_state,ADF_simple_next_state_weight] = dynamic_transition(ADF_simple_last_state,ADF_simple_current_max_d,transition_models);
        ADF_simple_measurement = measure(ADF_simple_next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            ADF_simple_next_state = fixed_states(t);
            ADF_simple_measurement = fixed_measurements(t);
        end
        %          ADF_next_state = discretized_next_state;
        ADF_simple_states=[ADF_simple_states;ADF_simple_next_state];
        %          ADF_measurement = discretized_measurement;
        ADF_simple_measurements=[ADF_simple_measurements;ADF_simple_measurement];
        ADF_simple_sampled_components = [ADF_simple_sampled_components;ADF_simple_next_state_weight];
    
        [pf_next_state,pf_next_state_weight] = dynamic_transition(pf_last_state,pf_current_max_d,transition_models);
        %          ADF_next_state = discretized_next_state;
        pf_measurement = measure(pf_next_state,measurement_model);
        if ~isempty(fixed_states_measurements)
            pf_next_state = fixed_states(t);
            pf_measurement = fixed_measurements(t);
        end
        pf_states=[pf_states;pf_next_state];
    
        %          ADF_measurement = discretized_measurement;
        pf_measurements=[pf_measurements;pf_measurement];
        pf_sampled_components = [pf_sampled_components;pf_next_state_weight];
    
        state_upper_check = discretized_next_state<state_range(2) && next_state<state_range(2)&&ADF_next_state<state_range(2)&&ADF_simple_next_state<state_range(2)&&ADF_gmm_unc_next_state<state_range(2)&&pf_next_state<state_range(2);
        state_lower_check = discretized_next_state>state_range(1) && next_state>state_range(1)&&ADF_next_state>state_range(1)&&ADF_simple_next_state>state_range(1)&&ADF_gmm_unc_next_state>state_range(1)&&pf_next_state>state_range(1);
        state_check = state_upper_check&&state_lower_check;
        if ~state_check
            return
        end
        %  the end of simulation
    
    
        last_mean_cov = initial_model;
        ADF_simple_last_mean_cov = initial_model;
        if t~=1
            last_message = forward_messages{t-1};
            last_mean_cov = {last_message{2}\last_message{1},inv(last_message{2})};
    
            ADF_simple_last_message = ADF_simple_forward_messages{t-1};
            ADF_simple_last_mean_cov = {ADF_simple_last_message{2}\ADF_simple_last_message{1},inv(ADF_simple_last_message{2})};
        end
    
        model = transition_models{current_max_d};
        [forward_mean_cov_new]=update_after_measurement(model,measurement_model,measurement,last_mean_cov);
        forward_messages{t}={forward_mean_cov_new{2}\forward_mean_cov_new{1},inv(forward_mean_cov_new{2})};
    
        ADF_model = transition_models{ADF_current_max_d};
        %          ADF_y_means = H*ADF_m_s+b;
        %          ADF_y_vars = R+H*ADF_v_s.*H';
        ADF_y_means = ADF_y_M;
        ADF_y_vars = ADF_y_V;
        ADF_weights_new = ADF_last_message{1}.*normpdf(ADF_measurement,ADF_y_means,sqrt(ADF_y_vars));
        ADF_weights_new = ADF_weights_new./sum(ADF_weights_new);
        %          ADF_post_vars = 1./(1./ADF_v_s+H'/R*H);
        %          ADF_post_means = ADF_post_vars.*(H'/R*(ADF_measurement-b)+1./ADF_v_s.*ADF_m_s);
        ADF_post_means = [];
        ADF_post_vars = [];
        for i=1:num_of_comp
            updated_mean_cov = update_after_measurement(ADF_model,measurement_model,ADF_measurement,{ADF_last_message{2}(i),ADF_last_message{3}(i)});
            ADF_post_means = [ADF_post_means;updated_mean_cov{1}];
            ADF_post_vars = [ADF_post_vars;updated_mean_cov{2}];
        end
        ADF_forward_messages{t+1}={ADF_weights_new,ADF_post_means,ADF_post_vars};
    
        ADF_gmm_unc_model = transition_models{ADF_gmm_unc_current_max_d};
        ADF_gmm_unc_model_weights = ADF_gmm_unc_model{1};
        ADF_gmm_unc_last_means = ADF_gmm_unc_last_message{2};
        ADF_gmm_unc_last_vars = ADF_gmm_unc_last_message{3};
        updated_means = [];
        updated_vars = [];
        updated_weights = ADF_gmm_unc_last_message{1}.*normpdf(ADF_gmm_unc_measurement,ADF_gmm_unc_y_M,sqrt(ADF_gmm_unc_y_V));
        updated_weights = updated_weights./sum(updated_weights);
        for i=1:num_of_comp
            updated_mean_cov = update_after_measurement(ADF_gmm_unc_model,measurement_model,ADF_gmm_unc_measurement,{ADF_gmm_unc_last_means(i),ADF_gmm_unc_last_vars(i)});
    
            updated_means = [updated_means;updated_mean_cov{1}];
            updated_vars = [updated_vars;updated_mean_cov{2}];
        end
    
        ADF_gmm_unc_updated_message = {updated_weights,updated_means,updated_vars};
        ADF_gmm_unc_inference{t+1}=ADF_gmm_unc_updated_message;
    
        ADF_simple_model = transition_models{ADF_simple_current_max_d};
        [ADF_simple_forward_mean_cov_new]=update_after_measurement(ADF_simple_model,measurement_model,ADF_simple_measurement,ADF_simple_last_mean_cov);
        ADF_simple_forward_messages{t}={ADF_simple_forward_mean_cov_new{2}\ADF_simple_forward_mean_cov_new{1},inv(ADF_simple_forward_mean_cov_new{2})};
    
        measurement_mean_vec = discretized_states.*H+b;
        measurement_var_vec = repmat(R,[1,num_of_bins]);
    
        delta_state = discretized_states(2)-discretized_states(1);
        log_discretized_marginal_likelihood = log_normpdf(discretized_measurement,measurement_mean_vec,sqrt(measurement_var_vec));
        log_dis_pos_dist = log_discretized_marginal_likelihood+log(current_max_predictive_dist);
        log_dis_pos_dist = log_dis_pos_dist - max(log_dis_pos_dist);
        dis_pos_dist = exp(log_dis_pos_dist)./sum(exp(log_dis_pos_dist))./delta_state;
    
        %          if t==1
        %              x1 = discretized_states;
        %              x2 = discretized_measurements;
        %              for d=1:K
        %                  discredization_pred_joint = discretize_pred_dists{3}{d};
        %                  adf_simple_pred_joint = ADF_simple_pred_dists{d};
        %                  adf_gmm_pred_joint = ADF_gmm_pred_dists{d};
        %                  adf_gm = gmdistribution(adf_gmm_pred_joint{2},adf_gmm_pred_joint{3},adf_gmm_pred_joint{1});
        %                  [X1,X2] = meshgrid(x1,x2);
        %                  X = [X1(:) X2(:)];
        %                  y = mvnpdf(X,adf_simple_pred_joint{1},adf_simple_pred_joint{2});
        %                  y = reshape(y,length(x2),length(x1));
        %                  p_gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(adf_gm,[x0 y0]),x,y);
        %                  contour(x1,x2,y,'--r')
        %                  hold on
        %                  fcontour(p_gmPDF,[min(x1) max(x1) min(x2) max(x2)],'-',"LineColor","b","MeshDensity",200)
        %                  hold on
        %                  contour(x1,x2,discredization_pred_joint)
        %                  hold off
        %                  grid on
        %                  legend("Direct Moment-matching","Constraint Moment-matching","Original GMM")
        %                  xlabel('X','FontSize', 14)
        %                  ylabel('Y','FontSize', 14)
        %                  title("Moment-matching comparison",'FontSize', 14)
        %                  saveas(gcf,"experiment_results/greedy ep/online decisions/different_decisions/approximation_"+string(t)+",d="+string(d)+".png");
        %              end
        %          end
        %
        %          for d=1:K
        %              discrete_dists = {discretize_pred_dists{1}{d},discretize_pred_dists{2}{d},discretize_pred_dists{3}{d},discretized_marginal_likelihood};
        %              plot_discretization_predictions(discretized_state_dist,discretized_states,discretized_measurements,discrete_dists,dis_pos_dist,d,t);
        %          end
    
    
        discretized_state_dist = dis_pos_dist;
        discretized_dist_at_step{t}= dis_pos_dist;
    
    
    
        reweight_start = tic;
        pf_model_chosen = transition_models{pf_current_max_d};
        [samples_of_last,sample_weights]=reweight_particles(num_of_samples,num_of_comp,sample_weights,samples_of_last,pf_model_chosen,measurement_model,pf_measurement,effective_sample_threshold);
        reweight_telapsed = toc(reweight_start);
        pf_time(t) = pf_time(t)+reweight_telapsed;
    
        %         discretized_info_gain =  sum(-current_max_predictive_dist(current_max_predictive_dist~=0).*log(current_max_predictive_dist(current_max_predictive_dist~=0)).*delta_state)-sum(-discretized_state_dist(discretized_state_dist~=0).*log(discretized_state_dist(discretized_state_dist~=0)).*delta_state);
        %        if t==1
        %            discretized_information(t) = discretized_info_gain;
        %        else
        %            discretized_information(t) = discretized_information(t-1) + discretized_info_gain;
        %        end
    
        %          converged = false;
        %          ite_num = 0;
        %          forward_pass_skipped_intotal = 0;
        %          backward_pass_skipped_intotal = 0;
        EP_convergence_start = tic;
        [forward_messages,backward_messages,useless_messages,forward_pass_skipped_intotal,backward_pass_skipped_intotal,ite_num]=clgsdm_general(initial_model,transition_models,measurement_model,t,convergence_threshold,measurements,decision_results(1:t),forward_messages,backward_messages);
        %          while ~converged
        %             ite_num = ite_num+1;
        %             [forward_messages,forward_converged,forward_pass_skipped]=update_forward_message(decision_results,measurements,forward_messages,backward_messages,transition_models,initial_model,measurement_model,t,convergence_threshold,step_size);
        %             [backward_messages,backward_converged,backward_pass_skipped]=update_backward_message(decision_results,measurements,forward_messages,backward_messages,transition_models,measurement_model,t,convergence_threshold,step_size);
        %             if ite_num<100
        %                 if forward_converged&&backward_converged
        %                     converged = true;
        %                 end
        %             else
        %                 [current_smoothed_means,current_smoothed_variances]=compute_smoothed_moments(forward_messages,backward_messages,t);
        %                 if isempty(smoothed_means)
        %                     smoothed_means_diffs = current_smoothed_means;
        %                 else
        %                     smoothed_means_diffs = abs(current_smoothed_means-smoothed_means);
        %                 end
        %                 if isempty(smoothed_variances)
        %                     smoothed_variances_diffs = current_smoothed_variances;
        %                 else
        %                     smoothed_variances_diffs = abs(current_smoothed_variances-smoothed_variances);
        %                 end
        %                if max(smoothed_means_diffs)<convergence_threshold&&max(smoothed_variances_diffs)<convergence_threshold
        %                     converged = true;
        %                 end
        %                 smoothed_variances = current_smoothed_variances;
        %                 smoothed_means = current_smoothed_means;
        %             end
        %
        %
        %             forward_pass_skipped_intotal = forward_pass_skipped_intotal+forward_pass_skipped;
        %             backward_pass_skipped_intotal = backward_pass_skipped_intotal+backward_pass_skipped;
        %
        %             if ite_num>100000
        % %                 for j=1:t
        % %                     plot_EP_agianst_numerical(discretized_states,discretized_dist_at_step{j},smoothed_means(j),smoothed_variances(j),j);
        % %                 end
        %                 X = 1:t;
        %                 y1 = measurements;
        %                 y2 = states;
        %                 figure,
        %                 plot(X,y1)
        %                 hold on
        %                 plot(X,y2)
        %                 legend('measurements','states')
        %                 xlabel('Time')
        %                 ylabel('measurements/states')
        %                 figure_title = "states measurements when stuck";
        %                 title(figure_title)
        %                 hold off
        %                 saveas(gcf,'experiment_results/greedy ep/online decisions/random dynamic transitions/'+figure_title+'.png');
        % %
        % %                 disp(smoothed_means_diffs);
        % %                 disp(smoothed_variances_diffs);
        %                 disp(discretized_decisions);
        %                 disp(states);
        %                 disp(measurements);
        %                 return
        %             end
        %          end
        EP_convergence_telapsed = toc(EP_convergence_start);
        EP_time(t) = EP_telapsed+EP_convergence_telapsed;
    
        %          for j=1:t
        %              plot_EP_agianst_numerical(discretized_states,discretized_dist_at_step{j},smoothed_means(j),smoothed_variances(j),j);
        %          end
        %         fid = fopen('experiment_results/exp.csv','a');
        %         fprintf(fid,'%d,%d,%d,%d\n',t,ite_num,forward_pass_skipped_intotal,backward_pass_skipped_intotal);
        %         fclose(fid);
        pf_mean(t)=dot(sample_weights,samples_of_last(:,1));
    end
    %     [ADF_means,ADF_stds]=compute_mean_std_from_message(ADF_forward_messages,{},T);
    ADF_means = [];
    ADF_stds = [];
    [ADF_simple_means,ADF_simple_stds] =compute_mean_std_from_message(ADF_simple_forward_messages,{},T);
    ADF_gmm_unc_means = [];
    ADF_gmm_unc_stds = [];
    for t=1:T
        [ADF_mean,ADF_var] = compute_moments_of_gmm(ADF_forward_messages{t+1});
        ADF_means = [ADF_means,ADF_mean];
        ADF_stds = [ADF_stds,sqrt(ADF_var)];
    
        [ADF_gmm_unc_mean,ADF_gmm_unc_var] = compute_moments_of_gmm(ADF_gmm_unc_inference{t+1});
        ADF_gmm_unc_means = [ADF_gmm_unc_means,ADF_gmm_unc_mean];
        ADF_gmm_unc_stds = [ADF_gmm_unc_stds,sqrt(ADF_gmm_unc_var)];
    end
    [EP_means,EP_stds]=compute_mean_std_from_message(forward_messages,backward_messages,T);
    
    EP_results = {decision_results,states,measurements,EP_means,EP_stds,EP_time,EP_sampled_components};
    ADF_results = {ADF_decision_results,ADF_states,ADF_measurements,ADF_means,ADF_stds,ADF_time,ADF_forward_messages,ADF_sampled_components};
    ADF_gmm_unc_results = {ADF_gmm_unc_decisions,ADF_gmm_unc_states,ADF_gmm_unc_measurements,ADF_gmm_unc_means,ADF_gmm_unc_stds,ADF_gmm_unc_time,ADF_gmm_unc_inference,ADF_gmm_unc_sampled_components};
    ADF_simple_results = {ADF_simple_decision_results,ADF_simple_states,ADF_simple_measurements,ADF_simple_means,ADF_simple_stds,ADF_simple_time,ADF_simple_sampled_components};
    discretized_results = {discretized_decisions,states_using_discretization,measurements_using_discretization,discretized_states,discretized_measurements,discretize_time,discretized_measurement_dist,discretization_sampled_components,discretized_dist_at_step};
    pf_results = {pf_decision_results,pf_states,pf_measurements,pf_time,pf_mean,pf_sampled_components};
end

function [current_max_d,current_m_s,current_v_s,current_M,current_V,MI_estimations,NA_estimations,pred_joint_dists]=make_decision_constrained(last_message,transition_models,measurement_model,K,discretized_states,discretized_measurements,fixed_decision)
    current_max_d = 0;
    current_max = 0;
    current_m_s = [];
    current_v_s = [];
    current_M = 0;
    current_V = 0;
    last_weights = last_message{1};
    last_means = last_message{2};
    last_vars = last_message{3};
    num_of_comp = length(transition_models{1}{1});
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    MI_estimations = [];
    NA_estimations = [];
    pred_joint_dists = {};
    for d=1:K
        if fixed_decision==0||fixed_decision==d
            model = transition_models{d};
            model_weights = model{1};
            model_coef = model{2};
            model_means = model{3};
            model_vars = model{4};
            m_ij = model_coef*last_means'+model_means;
            M = H*sum(m_ij.*last_weights'.*model_weights,'all')+b;
    %         m_s = sum(m_ij.*last_weights',2);
    %         v_s = sum((P_ij+m_ij.*m_ij).*last_weights',2)-m_s.*m_s;
            P_ij = model_coef*last_vars'.*model_coef+model_vars;
            P = sum((R+H.*P_ij.*H'+(H.*m_ij+b).*(H.*m_ij+b)).*last_weights'.*model_weights,'all')-M*M';
            current_info_gain = 0;
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
                
                
                current_info_gain = current_info_gain+0.5*model_weights(j)*log(det(M_j+F_j*P*F_j')/det(M_j));
                joint_covs(:,:,j)=[M_j+F_j*P*F_j',F_j*P;P*F_j',P];
                joint_means(j,:)=[m_j,M_j];
            end
            
            MI_estimations = [MI_estimations,current_info_gain];
            pred_joint_dists{d} = {model_weights,joint_means,joint_covs}; 
    %         NA_estimation = compute_na_gmm_mi({model_weights',joint_means,joint_covs},discretized_states,discretized_measurements);
    %         NA_estimations = [NA_estimations,NA_estimation];
            if current_info_gain>current_max
              current_max=current_info_gain;
              current_max_d = d;
              current_m_s = m_s;
              current_v_s = v_s;
              current_M = M;
              current_V = P;
           end
        end
        
    end
end


function [current_max_d,MI_estimations,pred_joint_dists]=make_decision(last_mean,last_var,transition_models,measurement_model,K,fixed_decision)
    current_max_d = 0;
    current_max = 0;
    MI_estimations = [];
    pred_joint_dists = {};
    for d=1:K
        if fixed_decision==0||fixed_decision==d
            model = transition_models{d};
           joint_model=directly_compute_moment_matching(model,measurement_model,last_mean,last_var);
           [joint_mean,joint_cov]=compute_moments_of_gmm(joint_model);
           pred_joint_dists{d} = {joint_mean,joint_cov};
           likelihood_var = joint_cov(2,2);
           marginal_var = joint_cov(1,1);
           current_info_gain = 0.5*log(marginal_var)+ 0.5*log(likelihood_var)-0.5*log(det(joint_cov));
           MI_estimations = [MI_estimations,current_info_gain];
           if current_info_gain>current_max
              current_max=current_info_gain;
              current_max_d = d;
           end
        end
       
    end
    
end

function [current_max_d,current_max_predictive_dist,MI_estimations,pred_dists] = make_decision_discretized(state_dist,discretized_states,discretized_measurements,discretized_trans_dist_per_model,discretized_measurement_dist,num_of_bins,K,fixed_decision)
    current_max_d = 0;
    current_max = 0;
    current_max_predictive_dist = state_dist;
    delta_state = discretized_states(2)-discretized_states(1);
    delta_measurement = discretized_measurements(2)-discretized_measurements(1);
    MI_estimations = [];
    pred_state_dists = {};
    pred_likelihoods = {};
    pred_joints = {};
    for d=1:K
       if fixed_decision==0||fixed_decision==d
           state_trans_probs = discretized_trans_dist_per_model{d};
           [pred_state_dist]=compute_discretized_predictive(state_dist,state_trans_probs,num_of_bins,delta_state);
           pred_state_dists{d} = pred_state_dist;
           joint_dist = discretized_measurement_dist.*pred_state_dist;
           pred_joints{d} = joint_dist;
           measurement_mariginal = sum(joint_dist,2).*delta_state;
           pred_likelihoods{d} = measurement_mariginal;
    %        joint_dist = joint_dist./sum(joint_dist,'all');
           predictive_entropy = -sum(pred_state_dist(pred_state_dist~=0).*log(pred_state_dist(pred_state_dist~=0)).*delta_state);
           measurement_entropy = -sum(measurement_mariginal(measurement_mariginal~=0).*log(measurement_mariginal(measurement_mariginal~=0)).*delta_measurement);
           joint_entropy = -sum(joint_dist(joint_dist~=0).*(log(joint_dist(joint_dist~=0))).*delta_state.*delta_measurement,'all');
           current_info_gain = predictive_entropy+measurement_entropy-joint_entropy;
           MI_estimations = [MI_estimations,current_info_gain];
           if current_info_gain>current_max
              current_max=current_info_gain;
              current_max_d = d;
              current_max_predictive_dist = pred_state_dist;
           end
       end
    end
    pred_dists = {pred_state_dists,pred_likelihoods,pred_joints};
end






