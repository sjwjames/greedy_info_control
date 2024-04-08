function test_sssm()
 %% setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_list = [20];
    K_list = 5:5;
    D_list = 9:10;
    debug_flag = false;
    directory_prefix = "experiment_results/greedy ep/online decisions/sssm/";
    if debug_flag
        directory_prefix = "experiment_results/greedy ep/online decisions/sssm/debug/";
    end
    fixed_setting_flag = false;
    E = 5;
    if fixed_setting_flag
        E = 1;
    end
 %% setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for T = 20:20
        for K=K_list
            for D=D_list
                %   dimension of latent states
                    dim=1;
                    
                    result_file_directory = directory_prefix+"T="+string(T)+"/K="+string(K)+",D="+string(D)+"/";
                    if not(isfolder(result_file_directory))
                       mkdir(result_file_directory)
                    end
%                     model_file = "experiment_results/greedy ep/online decisions/sssm/"+"T="+string(T)+"/K="+string(K)+",D="+string(D)+"/sssm.mat";
%                     if isfile(model_file)
%                         load(model_file);
%                     else
%                         sssm=generate_sssm_model(K,D,dim);
%                         save(result_file_directory+"sssm.mat","sssm");
%                     end
                    sssm=generate_sssm_model(K,D,dim);
                    
                % measurement model
                    H=1;
                    measurement_noise_mu = 0;
                    measurement_noise_var = 1;
                    measurement_model = {H,measurement_noise_mu,measurement_noise_var};
                
                % initial discrete state
                    initial_s = randi([1,K]);
                    initial_state = 0;
                    initial_mean = 0; 
                    initial_var = 1;
                    initial_model = {initial_s,initial_state,1/K,initial_mean,initial_var};
                    
                
                    max_state = 0;
                    min_state = 0;
                    max_measurement = 0;
                    min_measurement = 0;
                    random_result_collections = {};
                % generate state and measurement range
                    num_of_iterations = 100;
                    random_steps = T;
                    state_minmax = [-inf,inf];
                    measurement_minmax = [-inf,inf];
                    
                    
                    state_range_limit = [-40,40];
                    trial_cnt = 1;
                    trial_cnt_limit = 100;
                    while (state_minmax(1)<state_range_limit(1)) || (state_minmax(2)>state_range_limit(2))
                        [state_minmax,measurement_minmax,random_result_collections]=generate_sssm_range(num_of_iterations,random_steps,initial_s,initial_state,D,sssm,measurement_model);
                        trial_cnt = trial_cnt+1;
                        if trial_cnt>trial_cnt_limit
                            if ~debug_flag
                                sssm=generate_sssm_model(K,D,dim);
                            end
                            trial_cnt = 1;
                        end
                    end
                
                    min_state = state_minmax(1);
                    max_state = state_minmax(2);
                    min_measurement = measurement_minmax(1);
                    max_measurement = measurement_minmax(2);  
                    num_of_bins = 3000;
                    state_range = linspace(state_range_limit(1),state_range_limit(2),num_of_bins);
                    measurement_range = linspace(state_range_limit(1),state_range_limit(2),num_of_bins) ;

                        
                    save(result_file_directory+"sssm_random_result_collections.mat","random_result_collections")
                %% experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    delta_X = state_range(2)-state_range(1);
                    delta_Y = measurement_range(2)-measurement_range(1);
                    state_probs = normpdf(state_range,initial_mean,sqrt(initial_var));
                    initial_discretized_state_probs = repmat(state_probs,[K,1]).*1/K;
                %   discretize the sssm 
                    x_transitions = sssm{2};
                    discretized_x_transitions = {};
                    for j=1:K
                        for d=1:D
                            A = x_transitions{j}{d}{1};
                            c = x_transitions{j}{d}{2};
                            Q = x_transitions{j}{d}{3};
%                             discretized dynamic transition, X_{t-1} x X_{t}. e.g., probs(i,j) = p(X_t = state_range(j)|X_{t-1} = state_range(i))
                            probs = zeros([num_of_bins,num_of_bins]);
                            for i=1:num_of_bins
                                probs(i,:) = normpdf(state_range,A*state_range(i)+c,sqrt(Q));
                            end
%                             probs = normpdf(repelem(state_range,num_of_bins),repmat(A*state_range+c,[1,num_of_bins]),repmat(repmat(sqrt(Q),[1,num_of_bins]),[1,num_of_bins]));
%                             discretized_x_transitions{j}{d} = reshape(probs,[num_of_bins,num_of_bins]);
                            discretized_x_transitions{j}{d} = probs;
                        end
                    end
                %   discretize the measurement model, measurement_probs(i,j) = p(Y_t = measurement_range(j)|X_t = state_range(i))
                    measurement_probs = zeros([num_of_bins,num_of_bins]);
                    for i = 1:num_of_bins
                        measurement_probs(i,:) = normpdf(measurement_range,H*state_range(i)+measurement_noise_mu,sqrt(measurement_noise_var));
                    end
                    
                    discretized_measurement_model=measurement_probs;
                    discretization_values = {num_of_bins,state_range,measurement_range,{sssm{1},discretized_x_transitions},discretized_measurement_model,initial_discretized_state_probs};
                %% experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    num_of_samples =1000;
                    
                    ADF_result_collections = {};
                    discretization_result_collections = {};
                    pf_result_collections = {};
%                     discretization_MIs = [];
                    e = 1;
                    fixed_setting = random_result_collections{1};
                    while e<=E
                        if fixed_setting_flag
                            [ADF_results,discretization_results,pf_results,state_check]=clgsdm_sssm(initial_model,sssm,measurement_model,T,K,D,dim,discretization_values,num_of_samples,fixed_setting); 
                        else
                            [ADF_results,discretization_results,pf_results,state_check]=clgsdm_sssm(initial_model,sssm,measurement_model,T,K,D,dim,discretization_values,num_of_samples,{}); 
                        end
                        if ~state_check
                            disp("Fall out of the range")
                            continue
                        end
                        ADF_result_collections{e} = ADF_results;
                        pf_result_collections{e} = pf_results;
                        discretization_result_collections{e} = discretization_results;
%                         discretization_MIs = [discretization_MIs,discretization_results{6}];
                        e = e+1;
                    end
                
                %% MI evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    ADF_MIs = [];
                    random_info_results = [];
                    pf_MIs = [];
                    discretization_MIs = [];
                    ADF_discretized_message_collection = {};
                    pf_discretized_message_collection = {};
                    ADF_MI_percentages = [];
                    pf_MI_percentages = [];
                    random_MI_precentages = [];
                    X = 1:T;
                    for e=1:E
                        ADF_results = ADF_result_collections{e};
                        [MIs,ADF_discretized_messages]=evaluate_MI(ADF_results{1},ADF_results{4},discretization_values,measurement_model,T,K);
                        ADF_MIs = [ADF_MIs,MIs];
                        ADF_discretized_message_collection{e} = ADF_discretized_messages;
                        
                        pf_results = pf_result_collections{e};
                        [pf_MI,pf_discretized_messages]=evaluate_MI(pf_results{1},pf_results{4},discretization_values,measurement_model,T,K);
                        pf_MIs = [pf_MIs,pf_MI];
                        pf_discretized_message_collection{e} = pf_discretized_messages;
                
                        random_result = random_result_collections{e};
                        [randomMIs]=evaluate_MI(random_result{1},random_result{3},discretization_values,measurement_model,T,K);
                        random_info_results = [random_info_results,randomMIs];
                        random_result{end+1} = randomMIs;
                        random_result_collections{e} = random_result;

                        discretization_result = discretization_result_collections{e};
                        [discretization_MI] = evaluate_MI(discretization_result{1},discretization_result{4},discretization_values,measurement_model,T,K);
                        discretization_MIs = [discretization_MIs,discretization_MI];

                        ADF_MI_percentages = [ADF_MI_percentages,MIs./discretization_MI];
                        pf_MI_percentages = [pf_MI_percentages,pf_MI./discretization_MI];
                        random_MI_precentages = [random_MI_precentages,randomMIs./discretization_MI];
                        if debug_flag
%                             figure,plot(X,ADF_results{1},'-*','Color','r','LineWidth',1)
%                             hold on
%                             plot(X,pf_results{1},'-*','Color','black','LineWidth',1)
%                             hold on
%                             plot(X,discretization_result{1},'-*','Color','b','LineWidth',1)
%                             legend('ADF Decisions','PF Decisions','NA Decisions','FontSize', 12)
%                             title("Decision making",'FontSize', 14)
%                             hold off
%                             exportgraphics(gcf,result_file_directory+"Decision Making at Iteration_"+string(e)+'.pdf');

%                             figure,plot(X,MIs,'-*','Color','r','LineWidth',1)
%                             hold on
%                             plot(X,pf_MI,'-*','Color','black','LineWidth',1)
%                             hold on
%                             plot(X,discretization_MI,'-*','Color','b','LineWidth',1)
%                             legend('ADF MI','PF MI','NA MI','FontSize', 12)
%                             title("MI per run",'FontSize', 14)
%                             hold off
%                             exportgraphics(gcf,result_file_directory+"MI at Iteration_"+string(e)+'.pdf');
                        end
                    end
                    
                    
                    
                %% plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    ADF_color = '#D95319';
                    pf_color = '#EDB120';
                    discretization_color='#FFFF00';
                    
                    for e=1:E
                        
                        discretization_results = discretization_result_collections{e};
                        dis_pdf_ys = zeros([T,num_of_bins]);
                        for t=1:T
                            forward_message = discretization_results{5}{t+1};
                            dis_pdf_ys(t,:) = sum(forward_message,1);
                        end
                        dis_file_name = result_file_directory + "discretization_inference_"+string(num_of_bins)+"_trial_"+string(e)+".pdf";
                        dis_means = sum(dis_pdf_ys.*state_range.*delta_X,2);
                        disc_result_collection = {discretization_results,dis_pdf_ys,dis_means,discretization_color,"Numerical Approximation"};
                        if ~fixed_setting_flag
                            plot_inference(discretization_results,state_range,dis_pdf_ys,T,dis_file_name,discretization_color,delta_X,dis_means,[]);
                        end
                        
                        
                    
                        ADF_results = ADF_result_collections{e};
                        adf_messages = ADF_results{5};
                        adf_pdf_x = state_range;
                        adf_pdf_ys = zeros([T,length(adf_pdf_x)]);
                        adf_means = zeros([T,1]);
                        for t=1:T
                            forward_message = ADF_discretized_message_collection{e}{t+1};
                            adf_pdf_ys(t,:) = sum(forward_message,1);
                            adf_forward_message = adf_messages{t+1};
                            ys = zeros([1,length(adf_pdf_x)]);
                            for j=1:K
                                pdf_j = makedist('Normal','mu',adf_forward_message{2}(j),'sigma',sqrt(adf_forward_message{3}(j)));
                                ys = ys+adf_forward_message{1}(j).*pdf(pdf_j,adf_pdf_x);
                            end
                            adf_pdf_ys(t,:) = ys;
                            adf_means(t) = sum(ys.*adf_pdf_x.*delta_X);
                        end
                        adf_result_collection = {ADF_results,adf_pdf_ys,adf_means,ADF_color,"ADF-GMM"};
                        adf_file_name = result_file_directory + "adf_inference_"+string(num_of_bins)+"_trial_"+string(e)+".pdf";
                        if ~fixed_setting_flag
                            plot_inference(ADF_results,adf_pdf_x,adf_pdf_ys,T,adf_file_name,ADF_color,delta_X,adf_means,[]);
                        end
                        
                
                
                        pf_results = pf_result_collections{e};
                        pf_pdf_x = state_range;
                        pf_pdf_ys = zeros([T,num_of_bins]);
                        pf_means = zeros([T,1]);
                        particles = zeros([T*num_of_samples,1]);
                        for t=1:T
                            forward_message = pf_discretized_message_collection{e}{t+1};
                            pf_pdf_ys(t,:) = sum(forward_message,1);
                            samples = pf_results{5}{t+1};
                            sample_weights = samples{1};
                            sample_means = samples{3};
                            sample_vars = samples{4};
                            sample_states = samples{5};
                %             ys = zeros([1,num_of_bins]);
                %             ys = zeros([1,length(pdf_x)]);
                %             for j=1:num_of_samples
                %                 pdf_j = makedist('Normal','mu',sample_means(j),'sigma',sqrt(sample_vars(j)));
                %                 ys = ys+sample_weights(j).*pdf(pdf_j,pdf_x);
                %             end
                            pf_means(t) = sample_weights*sample_states';
                            particles((t-1)*num_of_samples+1:t*num_of_samples)= sample_states;
                %             pdf_ys(t,:) = ys;
                        end
                        pf_result_collection = {pf_results,pf_pdf_ys,pf_means,pf_color,"PF",particles};
                        pf_file_name = result_file_directory+"pf_inference_"+string(num_of_bins)+"_trial_"+string(e)+".pdf";
%                         if ~fixed_setting_flag
%                             plot_inference(pf_results,pf_pdf_x,pf_pdf_ys,T,pf_file_name,pf_color,delta_X,pf_means,particles);
%                         end
                        if fixed_setting_flag
                            plot_fixed_setting_inference(state_range,result_file_directory,{adf_result_collection,pf_result_collection,disc_result_collection});
                        end
                    end
                
                %% plot MI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    discretized_information_max = max(discretization_MIs,[],2);
                    discretized_information_min = min(discretization_MIs,[],2);
                    discretized_information_median = median(discretization_MIs,2);
                    discretized_information_mean = mean(discretization_MIs,2);
                    
                    ADF_information_percentage_median = median(ADF_MIs,2)./discretized_information_median;
                    random_information_percentage_median = median(random_info_results,2)./discretized_information_median;
                    pf_information_percentage_median = median(pf_MIs,2)./discretized_information_median;
                    
                    ADF_information_percentage_max = max(ADF_MIs,[],2)./discretized_information_max;
                    random_information_percentage_max = max(random_info_results,[],2)./discretized_information_max;
                    pf_information_percentage_max = max(pf_MIs,[],2)./discretized_information_max;
                    
                    ADF_information_percentage_min = min(ADF_MIs,[],2)./discretized_information_min;
                    random_information_percentage_min = min(random_info_results,[],2)./discretized_information_min;
                    pf_information_percentage_min = min(pf_MIs,[],2)./discretized_information_min;

                    ADF_MI_mean = mean(ADF_MIs,2);
                    PF_MI_mean = mean(pf_MIs,2);
                    NA_MI_mean = mean(discretization_MIs,2);

                    ADF_MI_std = std(ADF_MIs,0,2);
                    PF_MI_std = std(pf_MIs,0,2);
                    NA_MI_std = std(discretization_MIs,0,2);
                    
                    X=1:T;
                    figure,
                    errorbar(X,ADF_MI_mean,ADF_MI_std,'-*','Color','r','LineWidth',1)
                    hold on
                    errorbar(X,PF_MI_mean,PF_MI_std,'-*','Color','black','LineWidth',1)
                    hold on
                    errorbar(X,NA_MI_mean,NA_MI_std,'-*','Color','b','LineWidth',1)
                    legend('ADF MI','PF MI','NA MI','FontSize', 12)
                    title("Average MI Values",'FontSize', 14)
                    hold off
                    exportgraphics(gcf,result_file_directory+"Average MI Values.pdf");
                
                    ADF_information_percentage_mean = (mean(ADF_MIs,2)./discretized_information_max)';
                    random_information_percentage_mean = (mean(random_info_results,2)./discretized_information_max)';
                    pf_information_percentage_mean = (mean(pf_MIs,2)./discretized_information_max)';
                    
                    ADF_information_percentage_std = std(ADF_MIs'./discretized_information_max');
                    random_information_percentage_std = std(random_info_results'./discretized_information_max');
                    pf_information_percentage_std = std(pf_MIs'./discretized_information_max');

%                     ADF_information_percentage_median = median(ADF_MI_percentages,2);
%                     random_information_percentage_median = median(random_MI_precentages,2);
%                     pf_information_percentage_median = median(pf_MI_percentages,2);
%                 
%                     ADF_information_percentage_mean = (mean(ADF_MI_percentages,2))';
%                     random_information_percentage_mean = (mean(random_MI_precentages,2))';
%                     pf_information_percentage_mean = (mean(pf_MI_percentages,2))';
% %                     
%                     ADF_information_percentage_std=std(ADF_MI_percentages,1,2)';
%                     random_information_percentage_std=std(random_MI_precentages,1,2)';
%                     pf_information_percentage_std=std(pf_MI_percentages,1,2)';
                
                    figure_title = 'Averaged Cumulative Mutual Information,E='+string(E)+',bins='+string(num_of_bins);
                    X2 = [X,fliplr(X)];
                    ADF_inbetween = [ADF_information_percentage_mean+ADF_information_percentage_std,fliplr(ADF_information_percentage_mean-ADF_information_percentage_std)];
                    random_inbetween = [random_information_percentage_mean+random_information_percentage_std,fliplr(random_information_percentage_mean-random_information_percentage_std)];
                    pf_inbetween = [pf_information_percentage_mean+pf_information_percentage_std,fliplr(pf_information_percentage_mean-pf_information_percentage_std)];
                    
%                     ADF_inbetween = [ADF_information_percentage_max',fliplr(ADF_information_percentage_min')];
%                     random_inbetween = [random_information_percentage_max',fliplr(random_information_percentage_min')];
%                     pf_inbetween = [pf_information_percentage_max',fliplr(pf_information_percentage_min')];
                
                    figure,
                   
                    f1=fill(X2,ADF_inbetween, [1 0.94 0.94]);
                    hold on
                    f2=fill(X2,random_inbetween, [0.94 1 0.94 ]);
                    hold on
                    f3=fill(X2,pf_inbetween, [0 0 0]);
                    hold on                             
                    
                    plot(X,ADF_information_percentage_mean,'-*','Color','r','LineWidth',1)
                    hold on
                    plot(X,random_information_percentage_mean,'-*','Color','g','LineWidth',1)
                    hold on
                    plot(X,pf_information_percentage_mean,'-*','Color','black','LineWidth',1)
                    hold on
                    
                    alpha(f1,0.7)
                    alpha(f2,0.7)
                    alpha(f3,0.2)
                    legend('','','','ADF-GMM','Random','PF','Location','southeast','FontSize', 12)
                %     legend('ADF information std','Random information std','ADF','Random','Location','southeast','FontSize', 12)
                    xlabel('Timesteps','FontSize', 14)
                    ylabel('Realized MI / Optimal MI','FontSize', 14)
                    title("Cumulative MI Relative to Optimal",'FontSize', 14)
                    hold off
                    exportgraphics(gcf,result_file_directory+figure_title+'.pdf');
                    close all
            end
        end
        
    end

end

function plot_fixed_setting_inference(state_range,result_file_directory,results)
    n = length(results);
    na_result_coll = results{end};
    na_online_results = na_result_coll{1};
    pdf_ys = na_result_coll{2};
    xs =na_online_results{2};
    measurements =na_online_results{4};
    T = length(xs);
    line_width = 3;
    X=1:T;
    figure,
    colormap(gray);
    imagesc(1:T, state_range,1-pdf_ys');
    hold on

    
    plot(X,xs,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
    hold on
   
    scatter(X,measurements,14,'*','LineWidth',line_width)
    hold on
    legend_list = {'True States','Estimated States'};
    for i =1:n
        means = results{i}{3};
        color = results{i}{4};
        plot(X,means,'-x','MarkerSize',10,'color',color,'LineWidth',line_width)
        legend_list{end+1}=results{i}{5};
        hold on
        if results{i}{5} == "PF"
            particles = results{i}{6};
            if ~isempty(particles)
                num_of_samples = length(particles)/T;
                particle_X = repelem(X,num_of_samples);
                scatter(particle_X,particles,10,'filled')
                legend_list{end+1}="PF particles";
                hold on
            end
        end
    end
    
    % alpha(f1,0.7)
    % alpha(f2,0.7)

    set(gca,'YDir','normal')

    legend(legend_list,'FontSize', 16,'Location','northwest')
    % legend("ADF Variance","EP Variance",'True States','Measurements',"ADF means","EP means",'Location','northwest')

    title("SSSM Information Control Inference",'FontSize', 18)
    xlabel('Timesteps','FontSize', 18)
    ylabel('States','FontSize', 18)
    hold off
    exportgraphics(gcf,result_file_directory+"SSSM Information Control Inference.pdf");
end

function plot_inference(online_results,pdf_x,pdf_ys,T,file_name,color,delta_X,means,particles)
     xs = online_results{2};
     measurements = online_results{4};
     
%      means = sum(pdf_ys.*pdf_x.*delta_X,2);
       line_width = 3;
    
        X=1:T;
        figure,
        colormap(gray);
        imagesc(1:T, pdf_x,1-pdf_ys');
        hold on
        
        % f1 = fill(X2,inbetween, color_assigned);
        % hold on
        % f2 = fill(X2,EP_inbetween, [0.94 0.94 1]);
        % hold on
        plot(X,xs,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
        hold on
        plot(X,means,'-x','MarkerSize',10,'color',color,'LineWidth',line_width)
        hold on
        scatter(X,measurements,14,'*','LineWidth',line_width)
        hold on
        if ~isempty(particles)
            num_of_samples = length(particles)/T;
            particle_X = repelem(X,num_of_samples);
            scatter(particle_X,particles,10,'filled')
        end
        % alpha(f1,0.7)
        % alpha(f2,0.7)
        
        set(gca,'YDir','normal') 
    
        legend('True States','Estimated States','Measurements','FontSize', 16,'Location','northwest')
        % legend("ADF Variance","EP Variance",'True States','Measurements',"ADF means","EP means",'Location','northwest')

        title("SSSM Information Control Inference",'FontSize', 18)
        xlabel('Timesteps','FontSize', 18)
        ylabel('States','FontSize', 18)
        hold off
        exportgraphics(gcf,file_name);
end


function [MIs,discretization_forward_messages]=evaluate_MI(decisions,measurements,discretization_values,measurement_model,T,K)
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
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    
    MIs = [];
    rhos = discretized_sssm{1};
    for t=1:T
        chosen_d = decisions(t);
        x_transitions = discretized_sssm{2};
        forward_dist = discretization_forward_messages{t};
        s_marginalized=rhos'*forward_dist;
        discretized_selected_augmented_dist={};
        for j=1:K
            predictive_marginal = sum(s_marginalized(j,:)'.*x_transitions{j}{chosen_d}.*delta_X,1)';
            joint_distribution = predictive_marginal.*discretized_measurement_model;
            discretized_selected_augmented_dist{j} = joint_distribution;
            
        end
       [forward_discretized_message,entropy_reduction_sum] = update_sssm_disc_dist(discretized_selected_augmented_dist,measurements(t),measurement_model,discretized_states,K,N,delta_X,delta_Y);

%         new_weights = weights;
%         if sum(marginal_likelihoods)~=0
%             log_weights = log(marginal_likelihoods);
%             log_weights = log_weights - max(log_weights);
%             new_weights = exp(log_weights)./sum(exp(log_weights));
%         end
%         weighted_posterior = sum(forward_discretized_message.*new_weights,1);
%         non_zero_predictive_marginal = predictive_marginals(predictive_marginals~=0);
%         non_zero_posterior = weighted_posterior(weighted_posterior~=0);
%         entropy_reduction_sum = -sum(non_zero_predictive_marginal.*log(non_zero_predictive_marginal).*delta_X)+sum(non_zero_posterior.*log(non_zero_posterior).*delta_X);
        discretization_forward_messages{t+1} = forward_discretized_message;
        if t==1
            MIs = [MIs;entropy_reduction_sum];
        else
            MIs = [MIs;entropy_reduction_sum+MIs(t-1)];
        end
    end
    
end

function [state_range,measurement_range,random_result_collections]=generate_sssm_range(num_of_iterations,time_steps,initial_s,initial_state,D,sssm,measurement_model)
    max_state = 0;
    min_state = 0;
    max_measurement = 0;
    min_measurement = 0;
    random_result_collections = {};
% generate state and measurement range
    for i = 1:num_of_iterations
        random_steps = time_steps;
        s_states = zeros([random_steps,1]);
        states = zeros([random_steps,1]);
        measurements = zeros([random_steps,1]);
        current_s = initial_s;
        current_x = initial_state;
        decisions = [];
        for t=1:random_steps
            decision = randi([1,D]);
            [s,x]=process_sssm_transition(current_s,current_x,decision,sssm);
            measurement = measure(x,measurement_model);
            current_s = s;
            current_x = x;
            s_states(t) = s;
            states(t) = current_x;
            measurements(t) = measurement;
            decisions = [decisions;decision];
        end
        random_result_collections{i}={decisions,s_states,states,measurements};
        min_state = min(min_state,min(states));
        max_state = max(max_state,max(states));
        min_measurement = min(min_measurement,min(measurements));
        max_measurement = max(max_measurement,max(measurements));
        
    end
    state_range=[min_state,max_state];
    measurement_range=[min_measurement,max_measurement];
end