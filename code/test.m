function test()
%% test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_component_set = [16];
for i=1:length(n_component_set)
    test_GMM_constraint_mm(n_component_set(i));
end


%% setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(1);
debug_flag = false;
T = 20;
K = 5;
decision_dim = 1;
decisions = [1:K];
num_of_comp = 8;
result_file_directory = "experiment_results/greedy ep/online decisions/different_decisions/n="+string(num_of_comp)+",k="+string(K)+"/";
initial_state = 0;
initial_mean = 0; 
initial_var = 1;
initial_model = {initial_mean,initial_var};
model_file = result_file_directory+"transition_models.mat";
if not(isfolder(result_file_directory))
    mkdir(result_file_directory)
end
if isfile(model_file)
    load(model_file);
else
    transition_models = generate_transition_model(K,1,2,num_of_comp,1,"");
    save(result_file_directory+"transition_models.mat","transition_models");
end
H=1;
measurement_noise_mu = 0;
measurement_noise_var = 2;
measurement_model = {H,measurement_noise_mu,measurement_noise_var};

convergence_threshold=0.0001;
discretize_bins = 6000;

 state_range_limit = [-20,20];
 state_range = [-inf,inf];
 measurement_range = [-inf,inf];
 trial_cnt_limit = 100;

  while (state_range(1)<state_range_limit(1)) || (state_range(2)>state_range_limit(2))
        state_range = [inf,-inf];
        measurement_range = [inf,-inf];
        for i = 1:trial_cnt_limit
            [cur_state_range,cur_measurement_range]=generate_joint_ranges(K,initial_state,transition_models,measurement_model,T*10);
            state_range = [min(state_range(1),cur_state_range(1)),max(state_range(2),cur_state_range(2))];
            measurement_range = [min(measurement_range(1),cur_measurement_range(1)),max(measurement_range(2),cur_measurement_range(2))];
        end
        if ~debug_flag
            transition_models = generate_transition_model(K,1,1,num_of_comp,1,"");
        end
   end

fixed_setting=generate_random_fixed_setting(T,initial_state,K,transition_models,measurement_model);
save(result_file_directory+"fixed_setting.mat","fixed_setting");
state_range = [state_range(1),state_range(2)];
measurement_range = [measurement_range(1),measurement_range(2)];
state_discretization = linspace(state_range(1),state_range(2),discretize_bins);
measurements_discretization = linspace(measurement_range(1),measurement_range(2),discretize_bins);
%% MC closed optimal offline training
N_particles = 1000;
N_beliefs = 200;
D = K;
K_nearest = 5;
distance_threshold = 0.1;
epoch_num = 11;
MC_POMDP_file_dir = "experiment_results/MCPOMDP/n="+string(num_of_comp)+",k="+string(K)+"/";
[q_table,mc_decision_collections,mc_states_collections,mc_measurement_collections,mc_bf_collections] = mc_closed_optimal(N_particles,T,D,K_nearest,N_beliefs,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num,{},state_range,MC_POMDP_file_dir);
%% MC closed optimal
if not(isfolder(MC_POMDP_file_dir))
   mkdir(MC_POMDP_file_dir)
end
epoch_num = 11;
% [q_table,mc_decision_collections,mc_states_collections,mc_measurement_collections,mc_bf_collections] = mc_closed_optimal(N_particles,T,D,K_nearest,N_beliefs,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num,q_table,state_range);

MC_info_results = zeros([epoch_num,T]);
for j=1:epoch_num
     mc_states = mc_states_collections(:,j);
     mc_measurements = mc_measurement_collections(:,j);
     mc_decisions = mc_decision_collections(j,:);
     mc_bf_states = mc_bf_collections{j};
     [MC_state_dist_at_steps,MC_discretized_states,MC_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[mc_states,mc_measurements],mc_decisions,state_discretization,measurements_discretization,{});
     MC_info_results(j,:) = MC_information;
     plot_clo_inference(mc_states,mc_measurements,MC_discretized_states,MC_state_dist_at_steps,mc_bf_states,MC_POMDP_file_dir+"N="+string(N_beliefs)+",j="+string(j)+" K_nearest_inference.pdf",T);

end
% Legends = {};
R = 1:epoch_num;
% for i=1:T
%     plot(R,MC_info_results(:,i),'LineWidth',1)
%     Legends{end+1}="t="+string(i);
%     hold on
% end
% legend(Legends)
plot(R,MC_info_results(:,T),'LineWidth',1)
legend("t="+string(T))
xlabel('Runs','FontSize', 14)
ylabel('Cumulative MI','FontSize', 14)
title("Cumulative MI",'FontSize', 14)
hold off
saveas(gcf,MC_POMDP_file_dir+'MC_MI_change.pdf');
%% MC closed optimal online decision-making
epoch_num = 11;
[q_table,mc_decision_collections,mc_states_collections,mc_measurement_collections,mc_bf_collections] = mc_closed_optimal(N_particles,T,D,K_nearest,N_beliefs,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num,q_table,state_range);

X=1:T;
MC_info_results = zeros([epoch_num,T]);
for j=1:epoch_num
     mc_states = mc_states_collections(:,j);
     mc_measurements = mc_measurement_collections(:,j);
     mc_decisions = mc_decision_collections(j,:);
     mc_bf_states = mc_bf_collections{j};
     [MC_state_dist_at_steps,MC_discretized_states,MC_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[mc_states,mc_measurements],mc_decisions,state_discretization,measurements_discretization,{});
     MC_info_results(j,:) = MC_information;
     plot_clo_inference(mc_states,mc_measurements,MC_discretized_states,MC_state_dist_at_steps,mc_bf_states,MC_POMDP_file_dir+"N="+string(N_beliefs)+",j="+string(j)+" K_nearest_inference.pdf",T);

end

%% Experiments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=1:T;
T=20; 
E=11;
EP_color = 'b';
ADF_color = 'r';
pf_color = 'y';
random_color = 'g';
ADF_gmm_unc_color = 'c';
ADF_simple_color = 'm';
na_color = [0.4940 0.1840 0.5560];
state_discretization = linspace(state_range(1),state_range(2),discretize_bins);
measurements_discretization = linspace(measurement_range(1),measurement_range(2),discretize_bins);
delta_state = state_discretization(2)-state_discretization(1);
prior_dist = normpdf(state_discretization,initial_mean,sqrt(initial_var));
% EP_experiment_results = zeros([length(discretize_bins),T]);
% ADF_experiment_results = zeros([length(discretize_bins),T]);
% EP_result_stds = zero([length(discretize_bins),1]);
% ADF_result_stds = zero([length(discretize_bins),1]);

warning('off','all')

    Ep_info_results = zeros([E,T]);
    ADF_info_results = zeros([E,T]);
    discretized_info_results = zeros([E,T]);
    pf_info_results = zeros([E,T]);
    random_info_results = zeros([E,T]);
    ADF_simple_info_results = zeros([E,T]);
    ADF_gmm_unc_info_results = zeros([E,T]);
    
    EP_time_result = zeros([E,T]);
    ADF_time_result = zeros([E,T]);
    discretize_time_result = zeros([E,T]);
    pf_time_result = zeros([E,T]);
    ADF_simple_time_result = zeros([E,T]);
    ADF_gmm_unc_time_result = zeros([E,T]);
    EP_decision_results = zeros([E,T]);
    ADF_decision_results = zeros([E,T]);
    ADF_simple_decision_results = zeros([E,T]);
    ADF_gmm_unc_decision_results = zeros([E,T]);
    discretization_decision_results = zeros([E,T]);
    random_decision_results = zeros([E,T]);
    pf_decision_results = zeros([E,T]);
    EP_RMSEs = zeros([E,1]);
    ADF_RMSEs = zeros([E,1]);
    PF_RMSEs = zeros([E,1]);
    EP_reslt_collection = {};
    ADF_reslt_collection = {};
    discretization_reslt_collection = {};
    random_reslt_collection = {};
    pf_reslt_collection = {};
    ADF_simple_reslt_collection={};
    ADF_gmm_unc_result_collection ={};
    j = 1;
    fixed_setting_flag = true;
    if fixed_setting_flag
        E = 1;
    end
    while j<=E
        if fixed_setting_flag
           [EP_results,ADF_results,discretized_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results,state_check]=clgsdm_GMM_discrete(initial_state,initial_model,transition_models,measurement_model,T,K,convergence_threshold,discretize_bins,state_range,measurement_range,fixed_setting);        
        else
           [EP_results,ADF_results,discretized_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results,state_check]=clgsdm_GMM_discrete(initial_state,initial_model,transition_models,measurement_model,T,K,convergence_threshold,discretize_bins,state_range,measurement_range,{});        
        end
        if ~state_check
            continue
        end
        EP_reslt_collection{j}=EP_results;
        ADF_reslt_collection{j}=ADF_results;
        discretization_reslt_collection{j}=discretized_results;
        random_reslt_collection{j}=random_results;
        pf_reslt_collection{j}=pf_results;
        ADF_simple_reslt_collection{j} = ADF_simple_results;
        ADF_gmm_unc_result_collection{j} = ADF_gmm_unc_results;
        EP_decisions = EP_results{1};
        EP_states = EP_results{2};
        EP_measurements = EP_results{3};
        EP_means = EP_results{4};
        EP_vars = EP_results{5};
        EP_time = EP_results{6};
        EP_time_result(j,:)=EP_time;
        EP_decision_results(j,:)=EP_decisions;
        
        ADF_decisions = ADF_results{1};
        ADF_states = ADF_results{2};
        ADF_measurements = ADF_results{3};
        ADF_time = ADF_results{6};
        ADF_time_result(j,:)=ADF_time;
        ADF_decision_results(j,:)=ADF_decisions;
        
        ADF_gmm_unc_decisions = ADF_gmm_unc_results{1};
        ADF_gmm_unc_states = ADF_gmm_unc_results{2};
        ADF_gmm_unc_measurements = ADF_gmm_unc_results{3};
        ADF_gmm_unc_time = ADF_gmm_unc_results{6};
        ADF_gmm_unc_time_result(j,:)=ADF_gmm_unc_time;
        ADF_gmm_unc_decision_results(j,:)=ADF_gmm_unc_decisions;
        
        discretization_decisions = discretized_results{1};
        discretization_states = discretized_results{2};
        discretization_measurements = discretized_results{3};

        state_discretization = discretized_results{4};
        measurements_discretization = discretized_results{5};
        discretize_time = discretized_results{6};
        discretized_measurement_dist = discretized_results{7};
        discretize_time_result(j,:)=discretize_time;
        discretization_decision_results(j,:)=discretization_decisions;

        random_decisions = random_results{1};
        random_states = random_results{2};
        random_measurements = random_results{3};
        random_decision_results(j,:)=random_decisions(1:T);

        pf_decisions = pf_results{1};
        pf_states = pf_results{2};
        pf_measurements = pf_results{3};
        pf_time_result(j,:)= pf_results{4};
        pf_decision_results(j,:)= pf_decisions;
        
        ADF_simple_decisions = ADF_simple_results{1};
        ADF_simple_decision_results(j,:)= ADF_simple_decisions;
        ADF_simple_states = ADF_simple_results{2};
        ADF_simple_measurements = ADF_simple_results{3};
        ADF_simple_time_result (j,:)= ADF_simple_results{6};
        ADF_simple_means = ADF_simple_results{4};
        ADF_simple_vars = ADF_simple_results{5};
%         
        [ADF_state_dist_at_steps,ADF_discretized_states,ADF_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[ADF_states,ADF_measurements],ADF_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [ADF_gmm_unc_state_dist_at_steps,ADF_gmm_unc_discretized_states,ADF_gmm_unc_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[ADF_gmm_unc_states,ADF_gmm_unc_measurements],ADF_gmm_unc_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [ADF_simple_state_dist_at_steps,ADF_simple_discretized_states,ADF_simple_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[ADF_simple_states,ADF_simple_measurements],ADF_simple_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [EP_state_dist_at_steps,EP_discretized_states,EP_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[EP_states,EP_measurements],EP_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [d_state_dist_at_steps,d_discretized_states,discretization_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[discretization_states,discretization_measurements],discretization_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [random_state_dist_at_steps,random_discretized_states,random_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[random_states,random_measurements],random_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
        [pf_state_dist_at_steps,pf_discretized_states,pf_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[pf_states,pf_measurements],pf_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);

        Ep_info_results(j,:)=EP_information;
        ADF_info_results(j,:)=ADF_information;
        discretized_info_results(j,:)=discretization_information;
        random_info_results(j,:)=random_information;
        pf_info_results(j,:)=pf_information;
        ADF_simple_info_results(j,:)=ADF_simple_information;
        ADF_gmm_unc_info_results(j,:) = ADF_gmm_unc_information;

%         figure,plot(1:T,EP_information,'b',X,random_information,'g',X,discretization_information,'r',X,pf_information,'y',X,ADF_simple_information,'c',X,ADF_gmm_unc_information,'m'),legend("EP","Random","Numerical","PF","ADF-Gaussian","ADF-GMM",'Location','northwest')
%         exportgraphics(gcf,result_file_directory+'MI_values_bins='+string(discretize_bins)+'_'+string(j)+'.pdf')

        EP_experiment_result = {EP_results{4},EP_results{5},EP_discretized_states,EP_state_dist_at_steps,EP_states,EP_measurements,"EP",EP_color};
        ADF_experiment_result = {ADF_results{4},ADF_results{5},ADF_discretized_states,ADF_state_dist_at_steps,ADF_states,ADF_measurements,"ADF-GMM Constrained Inference",ADF_color};
        ADF_simple_experiment_result = {ADF_simple_results{4},ADF_simple_results{5},ADF_simple_discretized_states,ADF_simple_state_dist_at_steps,ADF_simple_states,ADF_simple_measurements,"ADF-Gaussian",ADF_simple_color};
        ADF_gmm_unc_result = {ADF_gmm_unc_results{4},ADF_gmm_unc_results{5},ADF_gmm_unc_discretized_states,ADF_gmm_unc_state_dist_at_steps,ADF_gmm_unc_states,ADF_gmm_unc_measurements,"ADF-GMM Unconstrained Inference",ADF_gmm_unc_color};
        pf_experiment_result = {pf_results{5},[],pf_discretized_states,pf_state_dist_at_steps,pf_states,pf_measurements,"PF",pf_color};
        na_result = {d_state_dist_at_steps*d_discretized_states'.*delta_state,[],d_discretized_states,d_state_dist_at_steps,discretization_states,discretization_measurements,"Numerical",na_color};
        

        if fixed_setting_flag
            plot_fixed_setting_inference({EP_experiment_result,ADF_experiment_result,ADF_simple_experiment_result,ADF_gmm_unc_result,pf_experiment_result,na_result},result_file_directory);
        end
%         plot_result(T,EP_experiment_result,"Inference result of EP,T="+string(T)+",K=" +string(K)+",Iteration="+string(j),EP_color,result_file_directory,"northwest");
%         plot_result(T,ADF_experiment_result,"Inference result of ADF-GMM Constrained Inference,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_color,result_file_directory,"northwest");
%         plot_result(T,ADF_simple_experiment_result,"Inference result of ADF-Gaussian,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_color,result_file_directory);
%         plot_result(T,ADF_gmm_unc_result,"Inference result of ADF-Gaussian Unconstrained Inference,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_color,result_file_directory,"northwest");
%         plot_result(T,pf_experiment_result,"Inference result of PF,T="+string(T)+",K="+string(K)+",Iteration="+string(j),pf_color,result_file_directory);
%         plot_result(T,na_result,"Inference result of numerical method,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_simple_color,result_file_directory);

%          ADF_gmm_unc_inference = ADF_gmm_unc_results{7};
%         ADF_simple_posterior = prior_dist;
%         EP_posterior = prior_dist;
%         PF_posterior = prior_dist;
%         ADF_gmm_posterior = prior_dist;
%         ADF_last_state = initial_state;
%         EP_last_state = initial_state;
%         PF_last_state = initial_state;
%         ADF_gmm_last_state = initial_state;
%         
%         target_file_directory = result_file_directory+"r="+string(j)+"/";
%         if not(isfolder(target_file_directory))
%            mkdir(target_file_directory)
%         end
%         for t=1:T
%             save_fig_location = target_file_directory+"t="+string(t)+"/";
%             if not(isfolder(save_fig_location))
%                mkdir(save_fig_location)
%             end
%             ADF_simple_chosen_model = transition_models{ADF_simple_decisions(t)};
%             EP_chosen_model = transition_models{EP_decisions(t)};
% %             PF_chosen_model = transition_models{PF_decisions(t)};
%             ADF_gmm_chosen_model = transition_models{ADF_gmm_unc_decisions(t)};
%             ADF_simple_prior = ADF_simple_posterior;
%             EP_prior = EP_posterior;
% %             PF_prior = PF_posterior;
%             ADF_gmm_prior = ADF_gmm_posterior;
%             ADF_exact_trainsition_props = evaluate_gmm({ADF_simple_chosen_model{1},ADF_simple_chosen_model{2}.*ADF_last_state+ADF_simple_chosen_model{3},ADF_simple_chosen_model{4}},state_discretization);
%             EP_exact_trainsition_props = evaluate_gmm({EP_chosen_model{1},EP_chosen_model{2}.*EP_last_state+EP_chosen_model{3},EP_chosen_model{4}},state_discretization);
% %             PF_exact_trainsition_props = evaluate_gmm({PF_chosen_model{1},PF_chosen_model{2}.*PF_last_state+PF_chosen_model{3},PF_chosen_model{4}},state_discretization);
%             ADF_gmm_exact_trainsition_props = evaluate_gmm({ADF_gmm_chosen_model{1},ADF_gmm_chosen_model{2}.*ADF_gmm_last_state+ADF_gmm_chosen_model{3},ADF_gmm_chosen_model{4}},state_discretization);
%             ADF_simple_posterior = normpdf(state_discretization,ADF_simple_means(t),sqrt(ADF_simple_vars(t)));
%             EP_posterior =  normpdf(state_discretization,EP_means(t),sqrt(EP_vars(t)));
%             ADF_gmm_posterior = evaluate_gmm(ADF_gmm_unc_inference{t+1},state_discretization);
%             ADF_simple_true_posterior = ADF_simple_state_dist_at_steps(t,:);
%             EP_true_posterior = EP_state_dist_at_steps(t,:);
% %             PF_true_posterior = pf_state_dist_at_steps(t,:);
%             ADF_gmm_true_posterior = ADF_gmm_unc_state_dist_at_steps(t,:);
%             figure,plot(state_discretization,ADF_simple_prior,'b','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_exact_trainsition_props,'g','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_simple_posterior,'m','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_simple_true_posterior,'r','LineWidth',1)
%             hold off
%             legend("ADF prior","Chosen dynamic transition model","ADF posterior projection","ADF exact posterior",'Location','northeast','FontSize', 12);
%             title("ADF Distributions");
%             exportgraphics(gcf,save_fig_location+"ADF distributions.pdf");
%             
%             figure,plot(state_discretization,EP_prior,'b','LineWidth',1)
%             hold on
%             plot(state_discretization,EP_exact_trainsition_props,'g','LineWidth',1)
%             hold on
%             plot(state_discretization,EP_posterior,'m','LineWidth',1)
%             hold on
%             plot(state_discretization,EP_true_posterior,'r','LineWidth',1)
%             hold off
%             legend("EP prior","Chosen dynamic transition model","EP posterior projection","EP exact posterior",'Location','northeast','FontSize', 12);
%             title("EP Distributions");
%             exportgraphics(gcf,save_fig_location+"EP distributions.pdf");
%             
%             figure,plot(state_discretization,ADF_gmm_prior,'b','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_gmm_exact_trainsition_props,'g','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_gmm_posterior,'m','LineWidth',1)
%             hold on
%             plot(state_discretization,ADF_gmm_true_posterior,'r','LineWidth',1)
%             hold off
%             legend("ADF-GMM prior","Chosen dynamic transition model","ADF-GMM posterior projection","ADF-GMM exact posterior",'Location','northeast','FontSize', 12);
%             title("ADF-GMM Distributions");
%             exportgraphics(gcf,save_fig_location+"ADF-GMM distributions.pdf");
%             ADF_last_state = ADF_simple_states(t);
%             EP_last_state = EP_states(t);
%             ADF_gmm_last_state = ADF_gmm_unc_states(t);
%             close all
%         end
        j = j + 1;
        close all
    end
%     writematrix(EP_decision_results,result_file_directory+'EP_decisions.csv');
%     writematrix(ADF_gmm_unc_decision_results,result_file_directory+'ADF_gmm_decisions.csv');
%     writematrix(ADF_simple_decision_results,result_file_directory+'ADF_gaussian_decisions.csv');
%     writematrix(pf_decision_results,result_file_directory+'PF_decisions.csv');
%     writematrix(discretization_decision_results,result_file_directory+'numerical_decisions.csv');
%     writematrix(EP_time_result,result_file_directory+'EP_running_time.csv');
%     writematrix(ADF_time_result,result_file_directory+'ADF_running_time.csv');
%     writematrix(ADF_gmm_unc_time_result,result_file_directory+'ADF_gmm_unc_running_time.csv');
%     writematrix(discretize_time_result,result_file_directory+'discretization_running_time.csv');
%     writematrix(pf_time_result,result_file_directory+'pf_running_time.csv');
%     writematrix(ADF_simple_time_result,result_file_directory+'ADF_simple_running_time.csv');
    %% temp code 
%     for j=1:E
%         pf_results = pf_reslt_collection{j};
%         pf_decisions = pf_results{1};
%         pf_states = pf_results{2};
%         pf_measurements = pf_results{3};
%         [pf_state_dist_at_steps,pf_discretized_states,pf_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[pf_states,pf_measurements],pf_decisions,state_discretization,measurements_discretization,discretized_measurement_dist);
%         pf_experiment_result = {pf_results{5},[],pf_discretized_states,pf_state_dist_at_steps,pf_states,pf_measurements,"PF"};
% 
%         
%         plot_result(T,pf_experiment_result,"Inference result of PF,T="+string(T)+",K="+string(K)+",Iteration="+string(j),pf_color,result_file_directory);
%     end
                plot_fixed_setting_inference({EP_experiment_result,ADF_experiment_result,ADF_simple_experiment_result,ADF_gmm_unc_result,pf_experiment_result,na_result},result_file_directory);

     
    %%
%     plot decisions
    X=1:T;
    for e=1:E
        figure,plot(X,EP_reslt_collection{e}{1},'-*','Color','r','LineWidth',1)
        hold on
        plot(X,discretization_reslt_collection{e}{1},'-*','Color','black','LineWidth',1)
        hold on
        plot(X,pf_reslt_collection{e}{1},'-*','Color','g','LineWidth',1)
        hold on
        plot(X,ADF_simple_reslt_collection{e}{1},'-*','Color','b','LineWidth',1)
        hold on
        plot(X,ADF_gmm_unc_result_collection{e}{1},'-*','Color','y','LineWidth',1)
        hold on
        plot(X,mc_decision_collections(e,:),'-*','Color','m','LineWidth',1)
        hold on
        legend('EP Decisions','NA Decisions','PF Decisions','ADF Gaussian Decisions','ADF GMM Decisions','MCPOMDP Decisions','FontSize', 12)
        title("Decision making",'FontSize', 14)
        hold off
        exportgraphics(gcf,result_file_directory+"Decision Making at Run_"+string(e)+'.pdf');
    end
    %%
%     EP_information_value_mean = mean(Ep_info_results);
%     ADF_information_value_mean = mean(ADF_info_results);
%     ADF_simple_information_value_mean = mean(ADF_simple_info_results);
%     ADF_gmm_unc_information_value_mean = mean(ADF_gmm_unc_info_results);
%     random_information_value_mean = mean(random_info_results);
%     pf_information_value_mean = mean(pf_info_results);
% %     mc_value_mean = mean(MC_info_results);
%     discretized_information_mean = mean(discretized_info_results);
% 
%     discretized_information_max = max(discretized_info_results);
%     EP_information_value_max = max(Ep_info_results);
%     ADF_information_value_max = max(ADF_info_results);
%     ADF_simple_information_value_max = max(ADF_simple_info_results);
%     ADF_gmm_unc_information_value_max = max(ADF_gmm_unc_info_results);
%     random_information_value_max = max(random_info_results);
%     pf_information_value_max = max(pf_info_results);
% %     mc_value_max = max(MC_info_results);
% 
%     discretized_information_min = min(discretized_info_results);
%     EP_information_value_min = min(Ep_info_results);
%     ADF_information_value_min = min(ADF_info_results);
%     ADF_simple_information_value_min = min(ADF_simple_info_results);
%     ADF_gmm_unc_information_value_min = min(ADF_gmm_unc_info_results);
%     random_information_value_min = min(random_info_results);
%     pf_information_value_min = min(pf_info_results);
% %     mc_value_min = min(MC_info_results);
%     
%     figure_title = 'Averaged Cumulative Mutual Information,E='+string(E)+',bins='+string(discretize_bins)+',steps='+string(T)+',K='+string(K);
%     X=1:T;
%     X2 = [X,fliplr(X)];
%     EP_inbetween = [EP_information_value_max,fliplr(EP_information_value_min)];
%     random_inbetween = [random_information_value_max,fliplr(random_information_value_min)];
%     pf_inbetween = [pf_information_value_max,fliplr(pf_information_value_min)];
% %     mc_inbetween = [mc_value_max,fliplr(mc_value_min)];
%     ADF_simple_inbetween = [ADF_simple_information_value_max,fliplr(ADF_simple_information_value_min)];
%     ADF_gmm_unc_inbetween = [ADF_gmm_unc_information_value_max,fliplr(ADF_gmm_unc_information_value_min)];
%     na_inbetween = [discretized_information_max,fliplr(discretized_information_min)];
% 
%     figure,
%     f1=fill(X2,EP_inbetween, 'b');
%     hold on
% %     f2=fill(X2,mc_inbetween, 'r');
% %     hold on
%     f3=fill(X2,random_inbetween, 'g');
%     hold on
%     f4=fill(X2,pf_inbetween, 'black');
%     hold on
%     f5 = fill(X2,ADF_gmm_unc_inbetween,'c');
%     hold on
%     f6 = fill(X2,ADF_simple_inbetween,'m');
%     hold on
%     f7 = fill(X2,ADF_simple_inbetween,'y');
%     hold on
%     plot(X,EP_information_value_mean,'-x','Color','b','LineWidth',1)
%     hold on
% %     plot(X,mc_value_mean,'-x','Color','r','LineWidth',1)
% %     hold on
%     plot(X,random_information_value_mean,'-x','Color','g','LineWidth',1)
%     hold on
%     plot(X,pf_information_value_mean,'-x','Color','black','LineWidth',1)
%     hold on
%     plot(X,ADF_gmm_unc_information_value_mean,'-x','Color','c','LineWidth',1)
%     hold on
%     plot(X,ADF_simple_information_value_mean,'-x','Color','m','LineWidth',1)
%     hold on
%     plot(X,discretized_information_mean,'-x','Color','y','LineWidth',1)
%     hold on
% 
% 
%     alpha(f1,0.2)
% %     alpha(f2,0.2)
%     alpha(f3,0.2)
%     alpha(f4,0.2)
%     alpha(f5,0.2)
%     alpha(f6,0.2)
%     alpha(f7,0.2)
%     legend('','','','','','','EP','Random','PF','ADF-GMM','ADF-Gaussian','NA','Location','northwest','FontSize', 12)
% %     legend('','','','','','','','EP','MCPOMDP','Random','PF','ADF-GMM','ADF-Gaussian','NA','Location','northwest','FontSize', 12)
%     xlabel('Timesteps','FontSize', 14)
%     ylabel('Realized MI','FontSize', 14)
%     title("Cumulative MI",'FontSize', 14)
%     hold off
%     exportgraphics(gcf,MC_POMDP_file_dir+figure_title+'.pdf');

    

    discretized_information_median = median(discretized_info_results);
    discretized_information_mean = mean(discretized_info_results);
    
    EP_information_percentage_median = median(Ep_info_results)./discretized_information_median;
    ADF_information_percentage_median = median(ADF_info_results)./discretized_information_median;
    ADF_simple_information_percentage_median = median(ADF_simple_info_results)./discretized_information_median;
    ADF_gmm_unc_information_percentage_median = median(ADF_gmm_unc_info_results)./discretized_information_median;
    random_information_percentage_median = median(random_info_results)./discretized_information_median;
    pf_information_percentage_median = median(pf_info_results)./discretized_information_median;
%     mc_information_percentage_median = median(MC_info_results)./discretized_information_median;
    
    EP_information_percentage_mean = mean(Ep_info_results)./discretized_information_mean;
    ADF_information_percentage_mean = mean(ADF_info_results)./discretized_information_mean;
    ADF_simple_information_percentage_mean = mean(ADF_simple_info_results)./discretized_information_mean;
    ADF_gmm_unc_information_percentage_mean = mean(ADF_gmm_unc_info_results)./discretized_information_mean;
    random_information_percentage_mean = mean(random_info_results)./discretized_information_mean;
    pf_information_percentage_mean = mean(pf_info_results)./discretized_information_mean;
%     mc_percentage_mean = mean(MC_info_results)./discretized_information_mean;

    EP_information_percentage_std=std(Ep_info_results./discretized_info_results);
    ADF_information_percentage_std=std(ADF_info_results./discretized_info_results);
    ADF_gmm_unc_information_percentage_std=std(ADF_gmm_unc_info_results./discretized_info_results);
    ADF_simple_information_percentage_std=std(ADF_simple_info_results./discretized_info_results);
    random_information_percentage_std=std(random_info_results./discretized_info_results);
    pf_information_percentage_std=std(pf_info_results./discretized_info_results);
%     mc_information_percentage_std=std(MC_info_results./discretized_info_results);
    
%     EP_information_percentage_best = max(Ep_info_results)./discretized_information_max;
%     EP_information_percentage_worst = min(Ep_info_results)./discretized_information_max;
% 
%     ADF_information_percentage_best = max(ADF_info_results)./discretized_information_max;
%     ADF_information_percentage_worst = min(ADF_info_results)./discretized_information_max;
%     
%     random_information_percentage_best = max(random_info_results)./discretized_information_max;
%     random_information_percentage_worst = min(random_info_results)./discretized_information_max;

%     mc_information_percentage_best = max(MC_info_results)./discretized_information_max;
%     mc_information_percentage_worst = min(MC_info_results)./discretized_information_max;
    
    figure_title = 'Averaged Cumulative Mutual Information,E='+string(E)+',bins='+string(discretize_bins)+',steps='+string(T)+',K='+string(K);
    X=1:T;
    X2 = [X,fliplr(X)];
    ADF_inbetween = [ADF_information_percentage_mean+ADF_information_percentage_std,fliplr(ADF_information_percentage_mean-ADF_information_percentage_std)];
    EP_inbetween = [EP_information_percentage_mean+EP_information_percentage_std,fliplr(EP_information_percentage_mean-EP_information_percentage_std)];
    random_inbetween = [random_information_percentage_mean+random_information_percentage_std,fliplr(random_information_percentage_mean-random_information_percentage_std)];
    pf_inbetween = [pf_information_percentage_mean+pf_information_percentage_std,fliplr(pf_information_percentage_mean-pf_information_percentage_std)];
%     mc_inbetween = [mc_percentage_mean+mc_information_percentage_std,fliplr(mc_percentage_mean-mc_information_percentage_std)];
    ADF_simple_inbetween = [ADF_simple_information_percentage_mean+ADF_simple_information_percentage_std,fliplr(ADF_simple_information_percentage_mean-ADF_simple_information_percentage_std)];
    ADF_gmm_unc_inbetween = [ADF_gmm_unc_information_percentage_mean+ADF_gmm_unc_information_percentage_std,fliplr(ADF_gmm_unc_information_percentage_mean-ADF_gmm_unc_information_percentage_std)];


    figure,
    f1=fill(X2,EP_inbetween, 'b');
    hold on
%     f2=fill(X2,mc_inbetween, 'r');
%     hold on
    f3=fill(X2,random_inbetween, 'g');
    hold on
    f4=fill(X2,pf_inbetween, 'black');
    hold on
    f5 = fill(X2,ADF_gmm_unc_inbetween,'c');
    hold on
    f6 = fill(X2,ADF_simple_inbetween,'m');
    hold on
    plot(X,EP_information_percentage_mean,'-x','Color','b','LineWidth',1)
    hold on
%     plot(X,mc_information_percentage_mean,'-x','Color','r','LineWidth',1)
%     hold on
    plot(X,random_information_percentage_mean,'-x','Color','g','LineWidth',1)
    hold on
    plot(X,pf_information_percentage_mean,'-x','Color','black','LineWidth',1)
    hold on
    plot(X,ADF_gmm_unc_information_percentage_mean,'-x','Color','c','LineWidth',1)
    hold on
    plot(X,ADF_simple_information_percentage_mean,'-x','Color','m','LineWidth',1)
    hold on
    alpha(f1,0.2)
%     alpha(f2,0.2)
    alpha(f3,0.2)
    alpha(f4,0.2)
    alpha(f5,0.2)
    alpha(f6,0.2)
    legend('','','','','','EP','Random','PF','ADF-GMM','ADF-Gaussian','Location','southwest','FontSize', 12)
%     legend('EP','ADF-GMM','Random','PF','ADF-GMM unconstrained','ADF-Gaussian','Location','southwest','FontSize', 12)
    xlabel('Timesteps','FontSize', 14)
    ylabel('Realized MI / Optimal MI','FontSize', 14)
    title("Cumulative MI Relative to Optimal",'FontSize', 14)
    hold off
    exportgraphics(gcf,result_file_directory+figure_title+'.pdf');    


end

function [x_pdfs]=evaluate_gmm(gmm_model,state_discretization)
    weights = gmm_model{1};
    means = gmm_model{2};
    vars = gmm_model{3};
    num_of_comp = length(weights);
    comp_pdfs = zeros([num_of_comp,length(state_discretization)]);
    for i =1:num_of_comp
        comp_pdfs(i,:) = normpdf(state_discretization,means(i),sqrt(vars(i)));
    end
    x_pdfs = weights'*comp_pdfs;
end

function fixed_setting=generate_random_fixed_setting(T,initial_state,K,transition_models,measurement_model)
    random_states = zeros([T,1]);
    random_measurements = zeros([T,1]);
    random_decisions = randi([1,K],1,T);
    for i=1:T
        last_state = initial_state;
        if i~=1
            last_state = random_states(i-1);
        end
        next_state = dynamic_transition(last_state,random_decisions(i),transition_models);
        measurement = measure(next_state,measurement_model);
        
    
        random_states(i)=next_state;
    
        random_measurements(i)=measurement;
    end
    
    fixed_setting = {[random_states,random_measurements],random_decisions};
end

function plot_fixed_setting_inference(result_collections,result_file_directory)

    na_experiment_collection = result_collections{end};

    discretized_bins = na_experiment_collection{3};
    state_dist_at_steps = na_experiment_collection{4};
    true_states = na_experiment_collection{5};
    measurements = na_experiment_collection{6};
    time_step = length(true_states);
    
    
    line_width = 3;

    n_methods = length(result_collections);
    X=1:time_step;

    figure,
    colormap(gray);
    imagesc(1:time_step, discretized_bins,1-state_dist_at_steps');
    hold on
    plot(X,true_states,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
    hold on
    scatter(X,measurements,14,'*','LineWidth',line_width)
    hold on
    legend_list = {'True States','Measurements'};
    for i =1:n_methods
        experiment_result = result_collections{i};
        legend_list{i+2} = experiment_result{7}+" estimation";
        plot(X,experiment_result{1},'--xr','MarkerSize',10,'LineWidth',line_width,'color',experiment_result{end})
        hold on
    end

    
    set(gca,'YDir','normal') 
    
    legend(legend_list,'Location',"northwest",'FontSize', 16)
    % legend("ADF Variance","EP Variance",'True States','Measurements',"ADF means","EP means",'Location','northwest')
    
    title("Inference for a fixed trajectory","FontSize",18)
    xlabel('Timesteps','FontSize', 18)
    ylabel('States','FontSize', 18)
    hold off
    exportgraphics(gcf,result_file_directory+"Inference for a fixed trajectory.pdf");
end


function plot_clo_inference(states,measurements,na_states,na_dist,belief_states,file_name,T)
     X=1:T;
     figure,
     colormap(gray);
     imagesc(1:T, na_states,1-na_dist');
     line_width = 3;
     hold on
     plot(X,states,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
     hold on
        
     scatter(X,measurements,14,'*','LineWidth',line_width)
     hold on
     if ~isempty(belief_states)
         particles = [];
         particle_sizes = [];
         means = [];
         for t=1:T
             belief_particle = belief_states{t+1}{2};
             belief_weights = belief_states{t+1}{1};
             N = length(belief_weights);
             particles = [particles;belief_particle];
             mean_val = dot(belief_weights,belief_particle);
             means = [means;mean_val];
             
             belief_weight_size = belief_weights.*N*10;
             
             
             particle_sizes = [particle_sizes;belief_weight_size];
         end
         plot(X,means,'-x','MarkerSize',10,'color','b','LineWidth',line_width)
         hold on
         particle_X = repelem(X,N);
         scatter(particle_X,particles,particle_sizes,'filled')
     end

        
        set(gca,'YDir','normal') 
    
        legend('True States','Measurements','Estimated States','Particles','FontSize', 16,'Location','northwest')

        title("MCPOMDP Information Control Inference",'FontSize', 18)
        xlabel('Timesteps','FontSize', 18)
        ylabel('States','FontSize', 18)
        hold off
        exportgraphics(gcf,file_name);
end