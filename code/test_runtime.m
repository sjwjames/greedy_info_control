function test_runtime()
    %% test runtime
    n_range = 2:8;
    k_range = 5:10;
    
    parfor num_of_comp=n_range
        for K=k_range
            if num_of_comp==2&&K==5
                continue
            end
            T = 20;
            decision_dim = 1;
            decisions = [1:K];
            result_file_directory = "experiment_results/greedy ep/online decisions/different_decisions/n="+string(num_of_comp)+",k="+string(K)+"/";
            if not(isfolder(result_file_directory))
                mkdir(result_file_directory)
            end
            
            initial_state = 0;
            initial_mean = 0; 
            initial_var = 1;
            initial_model = {initial_mean,initial_var};
            transition_models = generate_transition_model(K,1,1,num_of_comp,1,"");
            H=1;
            measurement_noise_mu = 0;
            measurement_noise_var = 1;
            measurement_model = {H,measurement_noise_mu,measurement_noise_var};
            
            convergence_threshold=0.0001;
            discretize_bins = 3000;
            
            state_range_limit = [-50,50];
            state_range = [-inf,inf];
            trial_cnt = 1;
            trial_cnt_limit = 30;
            while (state_range(1)<state_range_limit(1)) || (state_range(2)>state_range_limit(2))
                 
                [state_range,measurement_range]=generate_joint_ranges(K,initial_state,transition_models,measurement_model,1000);
                trial_cnt = trial_cnt+1;
                if trial_cnt>trial_cnt_limit
                    transition_models = generate_transition_model(K,1,1,num_of_comp,1,"");
                    trial_cnt = 1;
                end
            end
            
            
            
            state_range = [state_range(1)-10,state_range(2)+10];
            measurement_range = [measurement_range(1)-10,measurement_range(2)+10];
            state_discretization = linspace(state_range(1),state_range(2),discretize_bins);
            measurements_discretization = linspace(measurement_range(1),measurement_range(2),discretize_bins);
            epoch_num = 5;
        
            discretize_bins = 3000;
            T=20; 
            E=epoch_num;
            EP_color = '#0072BD';
            ADF_color = '#D95319';
            pf_color = '#EDB120';
            random_color = '#A2142F';
           
        
        
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
            while j<=E
                [EP_results,ADF_results,discretized_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results,state_check]=clgsdm_GMM_discrete(initial_state,initial_model,transition_models,measurement_model,T,K,convergence_threshold,discretize_bins,state_range,measurement_range,{});        
                disp(j)
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
                ADF_simple_states = ADF_simple_results{2};
                ADF_simple_measurements = ADF_simple_results{3};
                ADF_simple_time_result (j,:)= ADF_simple_results{6};
                disp(j)
                j = j+1;
            end
            writematrix(EP_time_result,result_file_directory+'EP_running_time.csv');
            writematrix(ADF_time_result,result_file_directory+'ADF_running_time.csv');
            writematrix(ADF_gmm_unc_time_result,result_file_directory+'ADF_gmm_unc_running_time.csv');
            writematrix(discretize_time_result,result_file_directory+'discretization_running_time.csv');
            writematrix(pf_time_result,result_file_directory+'pf_running_time.csv');
            writematrix(ADF_simple_time_result,result_file_directory+'ADF_simple_running_time.csv');
        end
    end
    
   
end