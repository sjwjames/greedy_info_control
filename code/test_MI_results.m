function test_MI_results()
    num_of_comps = 2:2:8;
    Ks = 6:7;
    for num_of_comp=num_of_comps
        for K=Ks
            T = 20;
            decision_dim = 1;
            decisions = [1:K];
            result_file_directory = "experiment_results/greedy ep/online decisions/different_decisions/n="+string(num_of_comp)+",k="+string(K)+"/";
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
            
             state_range_limit = [-30,30];
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
            
            
            save(result_file_directory+"transition_models.mat","transition_models");
            state_range = [state_range(1)-10,state_range(2)+10];
            measurement_range = [measurement_range(1)-10,measurement_range(2)+10];
            state_discretization = linspace(state_range(1),state_range(2),discretize_bins);
            measurements_discretization = linspace(measurement_range(1),measurement_range(2),discretize_bins);
            save(result_file_directory+"state_discretization.mat","state_discretization");
            save(result_file_directory+"measurements_discretization.mat","measurements_discretization")
            delta_state = state_discretization(2)-state_discretization(1);
            epoch_num = 5;
%             %% MC closed optimal
%             N = 100;
%             D = K;
%             K_nearest = 3;
%             distance_threshold = 0.1;
%             epoch_num = 5;
%             [q_table,mc_decision_collections,mc_states_collections,mc_measurement_collections] = mc_closed_optimal(N,T,D,K_nearest,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num);
%             %% MC closed optimal
%             
%             X=1:T;
%             Legends = {};
%             MC_info_results = zeros([epoch_num,T]);
%             figure,
%             for j=1:epoch_num
%                  mc_states = mc_states_collections(:,j);
%                  mc_measurements = mc_measurement_collections(:,j);
%                  mc_decisions = mc_decision_collections(j,:);
%                  [MC_state_dist_at_steps,MC_discretized_states,MC_information]=clgsdm_ra(initial_state,initial_model,transition_models,measurement_model,discretize_bins,T,K,[mc_states,mc_measurements],mc_decisions,state_discretization,measurements_discretization,{});
%                  MC_info_results(j,:) = MC_information;
%                  plot(X,MC_information,'-x','LineWidth',1)
%                  hold on
%                  Legends{end+1}="Epoch "+string(j);
%                  
%             end
%             legend(Legends)
%             xlabel('Timesteps','FontSize', 14)
%             ylabel('Accumulative MI','FontSize', 14)
%             title("Accumulative MI",'FontSize', 14)
%             hold off
            %% Experiments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            discretize_bins = 3000;
            T=20; 
            E=epoch_num;
            EP_color = 'b';
            ADF_color = 'r';
            pf_color = 'k';
            random_color = 'g';
            ADF_gmm_unc_color = 'c';
            ADF_simple_color = 'm';
            na_color = 'y';
        
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
                if ~state_check
                    continue
                end
        %         [EP_results,ADF_results,discretized_results,random_results,pf_results]=clgsdm_GMM_discrete(initial_state,initial_model,transition_models,measurement_model,T,K,convergence_threshold,discretize_bins,state_range,measurement_range,{states_measurements,prest_decisions});
        %         plot_result(T,states,measurements,forward_messages,backward_messages,ADF_messages,'states posterior',discretized_states,state_dist_at_steps);
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
                

                EP_experiment_result = {EP_results{4},EP_results{5},EP_discretized_states,EP_state_dist_at_steps,EP_states,EP_measurements,"EP"};
                ADF_experiment_result = {ADF_results{4},ADF_results{5},ADF_discretized_states,ADF_state_dist_at_steps,ADF_states,ADF_measurements,"ADF-GMM Constrained Inference"};
                ADF_simple_experiment_result = {ADF_simple_results{4},ADF_simple_results{5},ADF_simple_discretized_states,ADF_simple_state_dist_at_steps,ADF_simple_states,ADF_simple_measurements,"ADF-Gaussian"};
                ADF_gmm_unc_result = {ADF_gmm_unc_results{4},ADF_gmm_unc_results{5},ADF_gmm_unc_discretized_states,ADF_gmm_unc_state_dist_at_steps,ADF_gmm_unc_states,ADF_gmm_unc_measurements,"ADF-GMM Unconstrained Inference"};
                pf_experiment_result = {pf_results{5},[],pf_discretized_states,pf_state_dist_at_steps,pf_states,pf_measurements,"PF"};
                na_result = {d_state_dist_at_steps*d_discretized_states'.*delta_state,[],d_discretized_states,d_state_dist_at_steps,discretization_states,discretization_measurements,"Numerical"};

%                 plot_result(T,EP_experiment_result,"Inference result of EP,T="+string(T)+",K=" +string(K)+",Iteration="+string(j),EP_color,result_file_directory,"northwest");
%                 plot_result(T,ADF_experiment_result,"Inference result of ADF-GMM Constrained Inference,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_color,result_file_directory,"northwest");
%                 plot_result(T,ADF_simple_experiment_result,"Inference result of ADF-Gaussian,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_simple_color,result_file_directory,"northwest");
%                 plot_result(T,ADF_gmm_unc_result,"Inference result of ADF-GMM Unconstrained Inference,T="+string(T)+",K="+string(K)+",Iteration="+string(j),ADF_gmm_unc_color,result_file_directory,"northwest");
%                 plot_result(T,pf_experiment_result,"Inference result of PF,T="+string(T)+",K="+string(K)+",Iteration="+string(j),pf_color,result_file_directory,"northwest");
%                 plot_result(T,na_result,"Inference result of numerical method,T="+string(T)+",K="+string(K)+",Iteration="+string(j),na_color,result_file_directory,"northwest");
                j = j+1;
            end
       
           
           
            discretized_information_max = max(discretized_info_results);
            
            EP_information_percentage_median = mean(Ep_info_results)./discretized_information_max;
            ADF_information_percentage_median = mean(ADF_info_results)./discretized_information_max;
            ADF_simple_information_percentage_median = mean(ADF_simple_info_results)./discretized_information_max;
            ADF_gmm_unc_information_percentage_median = mean(ADF_gmm_unc_info_results)./discretized_information_max;
            random_information_percentage_median = mean(random_info_results)./discretized_information_max;
            pf_information_percentage_median = mean(pf_info_results)./discretized_information_max;
%             mc_information_percentage_median = median(MC_info_results)./discretized_information_max;
            
            EP_information_percentage_std=std(Ep_info_results./discretized_information_max);
            ADF_information_percentage_std=std(ADF_info_results./discretized_information_max);
            ADF_gmm_unc_information_percentage_std=std(ADF_gmm_unc_info_results./discretized_information_max);
            ADF_simple_information_percentage_std=std(ADF_simple_info_results./discretized_information_max);
            random_information_percentage_std=std(random_info_results./discretized_information_max);
            pf_information_percentage_std=std(pf_info_results./discretized_information_max);
            
        %     EP_information_percentage_best = max(Ep_info_results)./discretized_information_max;
        %     EP_information_percentage_worst = min(Ep_info_results)./discretized_information_max;
        % 
        %     ADF_information_percentage_best = max(ADF_info_results)./discretized_information_max;
        %     ADF_information_percentage_worst = min(ADF_info_results)./discretized_information_max;
        %     
        %     random_information_percentage_best = max(random_info_results)./discretized_information_max;
        %     random_information_percentage_worst = min(random_info_results)./discretized_information_max;
        
%             mc_information_percentage_best = max(MC_info_results)./discretized_information_max;
%             mc_information_percentage_worst = min(MC_info_results)./discretized_information_max;
            
            figure_title = 'Averaged Cumulative Mutual Information';
            X=1:T;
            X2 = [X,fliplr(X)];
            ADF_inbetween = [ADF_information_percentage_median+ADF_information_percentage_std,fliplr(ADF_information_percentage_median-ADF_information_percentage_std)];
            EP_inbetween = [EP_information_percentage_median+EP_information_percentage_std,fliplr(EP_information_percentage_median-EP_information_percentage_std)];
            random_inbetween = [random_information_percentage_median+random_information_percentage_std,fliplr(random_information_percentage_median-random_information_percentage_std)];
            pf_inbetween = [pf_information_percentage_median+pf_information_percentage_std,fliplr(pf_information_percentage_median-pf_information_percentage_std)];
        %     mc_inbetween = [mc_information_percentage_best,fliplr(mc_information_percentage_worst)];
            ADF_simple_inbetween = [ADF_simple_information_percentage_median+ADF_simple_information_percentage_std,fliplr(ADF_simple_information_percentage_median-ADF_simple_information_percentage_std)];
            ADF_gmm_unc_inbetween = [ADF_gmm_unc_information_percentage_median+ADF_gmm_unc_information_percentage_std,fliplr(ADF_gmm_unc_information_percentage_median-ADF_gmm_unc_information_percentage_std)];
        
        
            figure,
            f1=fill(X2,EP_inbetween, 'b');
            hold on
%             f2=fill(X2,ADF_inbetween, 'r');
%             hold on
            f3=fill(X2,random_inbetween, 'g');
            hold on
            f4=fill(X2,pf_inbetween, 'black');
            hold on
            f5 = fill(X2,ADF_gmm_unc_inbetween,'c');
            hold on
            f6 = fill(X2,ADF_simple_inbetween,'m');
            hold on
            plot(X,EP_information_percentage_median,'-x','Color','b','LineWidth',1)
            hold on
%             plot(X,ADF_information_percentage_median,'-x','Color','r','LineWidth',1)
%             hold on
            plot(X,random_information_percentage_median,'-x','Color','g','LineWidth',1)
            hold on
            plot(X,pf_information_percentage_median,'-x','Color','black','LineWidth',1)
            hold on
            plot(X,ADF_gmm_unc_information_percentage_median,'-x','Color','c','LineWidth',1)
            hold on
            plot(X,ADF_simple_information_percentage_median,'-x','Color','m','LineWidth',1)
            hold on
        
        %     errorbar(X,EP_information_percentage_median,EP_information_percentage_std,'-x','Color','b','LineWidth',1)
        %     hold on
        %     errorbar(X,ADF_information_percentage_median,ADF_information_percentage_std,'-x','Color','r','LineWidth',1)
        %     hold on
        %     errorbar(X,random_information_percentage_median,random_information_percentage_std,'-x','Color','g','LineWidth',1)
        %     hold on
        %     errorbar(X,pf_information_percentage_median,pf_information_percentage_std,'-x','Color','black','LineWidth',1)
        %     hold on
        %     errorbar(X,ADF_gmm_unc_information_percentage_median,ADF_gmm_unc_information_percentage_std,'-x','Color','c','LineWidth',1)
        %     hold on
        %     errorbar(X,ADF_simple_information_percentage_median,ADF_simple_information_percentage_std,'-x','Color','m','LineWidth',1)
        %     hold on
            alpha(f1,0.2)
%             alpha(f2,0.2)
            alpha(f3,0.2)
            alpha(f4,0.2)
            alpha(f5,0.2)
            alpha(f6,0.2)
            legend('','','','','','EP','Random','PF','ADF-GMM','ADF-Gaussian','Location','southwest','FontSize', 12)
        %     legend('EP','ADF-GMM','Random','PF','ADF-GMM unconstrained','ADF-Gaussian','Location','southwest','FontSize', 12)
            xlabel('Timesteps','FontSize', 14)
            ylabel('Accumulative MI(estimation)/Accumulative MI(exact)','FontSize', 14)
            title("Accumulative MI Percentage",'FontSize', 14)
            hold off
            exportgraphics(gcf,result_file_directory+figure_title+'.pdf');
            close all
        end
       
    end
end