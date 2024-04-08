function test_cpomdp()
%% Environment setup
    debug_flag = false;
    T = 50;
    K = 3;
    num_of_comp = 1;
    result_file_directory = "experiment_results/greedy ep/online decisions/cpomdp/n="+string(num_of_comp)+",k="+string(K)+"/";
    initial_state = randi([-20,20],1);
%     initial_state = 20;
    transition_models = generate_transition_model(K,1,2,num_of_comp,1,"CPOMDP");
    if not(isfolder(result_file_directory))
        mkdir(result_file_directory)
    end
    M=22;
    N=22;
    gS=cell(1,M);
    k=1;    
    so=1.6;
    for i=-21:2:21
       gS{k}={1/M,i,so};
       k=k+1;
    end
    indices = -21:1:21;
    k = 1;
    v = 1.6;
    initial_belief=cell(1,length(indices));
    for i=indices
       initial_belief{k}={1/length(indices),i,v};
       k=k+1;
    end
    initial_belief = gS;
    gl={1/N,1,0.05};
    gr={1/N,2,0.05};
    gc={1/N,3,0.05};
    gd={1/N,4,0.05};
    measurement_model={gS,{gl gl gl gl gl gd gc gc gd gc gc gc gd gc gc gd gc gr gr gr gr gr}};
    explicit_reward_model = {};
    explicit_reward_model{1} = {{-2,-21,1},{-2,-19,1},{-2,-17,1}};
    explicit_reward_model{2} = {{-2,21,1},{-2,19,1},{-2,17,1}};
    explicit_reward_model{3} = {{-10,-25,250},{2,3,3},{-10,25,250}};
    
    plot_initial_belief(initial_belief);
%% plot models
      plot_reward_model(explicit_reward_model,result_file_directory);
%       X = -20:0.01:20;
%       left_X = X-2;
%       right_X = X+2;
%       Xs = [left_X;right_X;X];
%       for i=1:length(explicit_reward_model)
%           reward_func = explicit_reward_model{i};
%           Ys_enter = cellfun(@(x)(x{1}.*normpdf(Xs(i,:),x{2},sqrt(x{3}))),reward_func,"UniformOutput",false);
%           Y_enter = zeros([1,length(X)]);
%           for j=1:length(Ys_enter)
%               Y_enter = Y_enter+Ys_enter{j};
%           end
%           ex_reward = sum(Y_enter./length(X));
%           disp(ex_reward)
%       end
    plot_obs_model(measurement_model,result_file_directory);
    plot_state_given_obs(measurement_model,[1,2,3,4],-21:0.05:21,result_file_directory);
%     plot_marginal_likelihood(measurement_model,result_file_directory);
%     plot_reward_model(explicit_reward_model,result_file_directory);
%     plot_obs_given_state(measurement_model,6,result_file_directory);
%     plot_obs_given_state(measurement_model,8,result_file_directory);
%     plot_obs_given_state(measurement_model,10,result_file_directory);
%     plot_obs_given_state(measurement_model,12,result_file_directory);
%     plot_obs_given_state(measurement_model,14,result_file_directory);
%     plot_obs_given_state(measurement_model,16,result_file_directory);
% %     plot_obs_given_state(measurement_model,-11,result_file_directory);
% %     plot_obs_given_state(measurement_model,-9,result_file_directory);
%     plot_marginal_likelihood(measurement_model,result_file_directory);
%     plot_state_given_obs(measurement_model,4,-21:0.05:21,result_file_directory);
%     plot_state_given_obs(measurement_model,0.4:0.2:1.6,-21:0.05:21,result_file_directory);
%     plot_state_given_obs(measurement_model,1.6:0.2:2.6,-21:0.05:21,result_file_directory);
%% Run C-POMDP experiment with 1-shot ADF greedy information control

    initial_states = -20:1:20;
    E = 11;
    run_flags = [false,false,false,false];
    if run_flags(1)
        results_collection = cell(length(initial_states),E);
    end
    if run_flags(2)
        ADF_ex_results_collection = cell(length(initial_states),E);
    end

    if run_flags(3)
        PF_results_collection = cell(length(initial_states),E);
    end

    if run_flags(4)
        PF_ex_results_collection = cell(length(initial_states),E);
    end
    
    
    
    
    debug_flag = true;
%     f(length(initial_states),E,4) = parallel.FevalFuture;
    for i = 1:length(initial_states)
        initial_state = initial_states(i);
        for e = 1:E
            %         initial_state = randi([-20,20],1);
%             f(i,e,:)=parfeval(@info_cpomdp,4,transition_models,measurement_model,initial_belief,initial_state,M,N,explicit_reward_model,T,[-25,25],{},true);
%             [ADF_results,PF_results,ADF_ex_results,PF_ex_results] = fetchOutputs(f(i,e,:));
            [ADF_results,PF_results,ADF_ex_results,PF_ex_results]=info_cpomdp(transition_models,measurement_model,initial_belief,initial_state,M,N,explicit_reward_model,T,[-25,25],{},true,run_flags);
           
%           
            if run_flags(1)
                results_collection{i,e}=ADF_results;
            end
            if run_flags(2)
                ADF_ex_results_collection{i,e} = ADF_ex_results;
            end
            
            if run_flags(3)
                PF_results_collection{i,e}=PF_results;
            end

            if run_flags(4)
                PF_ex_results_collection{i,e} = PF_ex_results;
            end
            
            
              %% Plot debug results

            if debug_flag

                if run_flags(1)
                    plot_cpomdp_results(ADF_results,M,T,K,e,result_file_directory+"mi_empowered/","ADF");
                end
                if run_flags(2)
                    plot_cpomdp_results(ADF_ex_results,M,T,K,e,result_file_directory+"ex_only/","ADF");
                end

                if run_flags(3)
                    plot_cpomdp_results(PF_results,M,T,K,e,result_file_directory+"PF_empowered/","PF");
                end

                if run_flags(4)
                    plot_cpomdp_results(PF_ex_results,M,T,K,e,result_file_directory+"PF_ex_only/","PF");
                end
                
                
                
                
            end

        end
        close all
    end
  %% temp
%     plot_cpomdp_results(ADF_results,M,T,K,e,result_file_directory+"mi_empowered/","ADF");
%     plot_cpomdp_results(PF_results,M,T,K,e,result_file_directory+"PF_empowered/","PF");
  %% load data and calculate stats
%     median_steps_per_start = zeros([length(initial_states),4]);
%     steps_std_per_start = zeros([length(initial_states),4]);
% 
%     median_final_reward_per_start = zeros([length(initial_states),4]);
%     final_reward_std_per_start = zeros([length(initial_states),4]);
%     worst_final_reward_per_start = zeros([length(initial_states),4]);
%     best_final_reward_per_start = zeros([length(initial_states),4]);
%     mean_final_reward_per_start = zeros([length(initial_states),4]);
%     for i=1:length(initial_states)
%         ADF_exact_explicit_rewards = zeros([E,1]);
%         ADF_ex_exact_explicit_rewards = zeros([E,1]);
%         ADF_steps = zeros([E,1]);
%         ADF_ex_steps = zeros([E,1]);
% 
%         PF_exact_explicit_rewards = zeros([E,1]);
%         PF_ex_exact_explicit_rewards = zeros([E,1]);
%         PF_steps = zeros([E,1]);
%         PF_ex_steps = zeros([E,1]);
%         for j=1:E
%             ADF_results = results_collection{i,j};
%             ADF_ex_results = ADF_ex_results_collection{i,j};
%             ADF_exact_explicit_rewards(j)= ADF_results{6}(end,3);
%             
%             ADF_ex_exact_explicit_rewards(j)= ADF_ex_results{6}(end,3);
%             ADF_steps(j)= length(ADF_results{2});
%             ADF_ex_steps(j)= length(ADF_ex_results{2});
%         end
% 
%         median_steps_per_start(i,:)=[median(ADF_steps),median(ADF_ex_steps)];
%         steps_std_per_start(i,:)=[std(ADF_steps),std(ADF_ex_steps)];
%         
%         median_final_reward_per_start(i,:)=[median(ADF_exact_explicit_rewards),median(ADF_ex_exact_explicit_rewards)];
%         mean_final_reward_per_start(i,:)=[mean(ADF_exact_explicit_rewards),mean(ADF_ex_exact_explicit_rewards)];
%         worst_final_reward_per_start(i,:)=[min(ADF_exact_explicit_rewards),min(ADF_ex_exact_explicit_rewards)];
%         best_final_reward_per_start(i,:)=[max(ADF_exact_explicit_rewards),max(ADF_ex_exact_explicit_rewards)];
%         final_reward_std_per_start(i,:)=[std(ADF_exact_explicit_rewards),std(ADF_ex_exact_explicit_rewards)];
%     end
    if isfile("cpomdp_results.mat")
        load("cpomdp_results.mat");
    else
        results_collections = {results_collection,ADF_ex_results_collection,PF_results_collection,PF_ex_results_collection};
    end
    

    if run_flags(1)
        results_collections{1} = results_collection;
    end
    if run_flags(2)
        results_collections{2} = ADF_ex_results_collection;
    end

    if run_flags(3)
        results_collections{3} = PF_results_collection;
    end

    if run_flags(4)
        results_collections{4} = PF_ex_results_collection;
    end
    
    [step_results,last_reward_results,total_reward_results,last_distance_results]=calculate_performance_stats(results_collections,initial_states,E);
    mean_final_reward_per_start = last_reward_results{2};
    final_reward_std_per_start = last_reward_results{end};

    median_final_distance_per_start = last_distance_results{1};
    mean_final_distance_per_start = last_distance_results{2};
    distance_std_per_start = last_distance_results{end};
    worst_final_distance_per_start = last_distance_results{3};
    best_final_distance_per_start = last_distance_results{4};
    
    median_total_reward_per_start = total_reward_results{1};
    mean_total_reward_per_start = total_reward_results{2};
    total_reward_std_per_start = total_reward_results{end};
    worst_total_reward_per_start = total_reward_results{3};
    best_total_reward_per_start = total_reward_results{4};

    median_steps_per_start = step_results{1};
    mean_steps_per_start = step_results{2};
    worst_steps_per_start = step_results{3};
    best_steps_per_start = step_results{4};
    steps_std_per_start = step_results{end};

    n_of_collections = length(results_collections);
    
    if ~isfile("cpomdp_results.mat")
        save "cpomdp_results.mat" results_collections
    end

%%     Plot final results
    colors = [0, 0.4470, 0.7410;0.8500, 0.3250, 0.0980;0.9290, 0.6940, 0.1250;0.4940, 0.1840, 0.5560];
    X=initial_states;
    X2 = [initial_states,fliplr(initial_states)];
    
    method_names = ["ADF MI-Empowered","ADF Explicit-Only","PF MI-Empowered","PF Explicit-Only"];
    legend_list = {};
    chosen_list = [1,3];
    figure,
    for j = 1:length(chosen_list)
%         errorbar(initial_states,mean_final_reward_per_start(:,i),final_reward_std_per_start(:,i),"Linewidth",1)
        i = chosen_list(j);
        inbetween = [mean_final_reward_per_start(:,i)'+final_reward_std_per_start(:,i)',fliplr(mean_final_reward_per_start(:,i)'-final_reward_std_per_start(:,i)')];
        f=fill(X2,inbetween, colors(i,:));
        legend_list{(j-1)*2+1}='';
        hold on
        alpha(f,0.2)
        plot(X,mean_final_reward_per_start(:,i),'-x','Color',colors(i,:),'LineWidth',3)
        legend_list{j*2}=method_names(i);
        hold on
    end
 
   
    legend(legend_list,'Location','best','FontSize', 14);
    xlabel('Initial Positions','FontSize', 16)
    ylabel('Mean Value of Reward at Final Step','FontSize', 16)
    title("Rewards at Final Step",'FontSize', 16)
     hold off
    exportgraphics(gcf,result_file_directory+"Rewards at Final Step.pdf");
    
%     bar chart with error bar template
%     ngroups = size(mean_final_distance_per_start, 2);
%     nbars = size(mean_final_distance_per_start, 1);
%     
%     b = bar(initial_states,mean_final_distance_per_start,'grouped');
%     x = nan(nbars, ngroups);
%     for i = 1:ngroups
%         x(:,i) = b(i).XEndPoints';
%     end
%     hold on
%     errorbar(x,mean_final_distance_per_start,distance_std_per_start,'k','linestyle','none');
%     hold off

    figure,
    for i = 1:n_of_collections
        errorbar(initial_states,mean_final_distance_per_start(:,i),zeros([length(initial_states),1]),distance_std_per_start(:,i),"Linewidth",3)
%         histogram(mean_final_distance_per_start(:,i),"Normalization","probability","BinWidth",0.25)
        hold on
    end
    
    
    hold off
    legend("ADF MI-Empowered","ADF Explicit-Only","PF MI-Empowered","PF Explicit-Only",'Location','best','FontSize', 14)
    xlabel('Initial Positions','FontSize', 16)
    ylabel('Distance to the correct door','FontSize', 16)
    title("Distance to the correct door at the last step",'FontSize', 16)
    exportgraphics(gcf,result_file_directory+"distance.pdf");


    figure,
    for i = 1:n_of_collections
        errorbar(initial_states,mean_steps_per_start(:,i),steps_std_per_start(:,i),"Linewidth",3)
        hold on
    end
    
    hold off
    legend("ADF MI-Empowered","ADF Explicit-Only","PF MI-Empowered","PF Explicit-Only",'Location','best','FontSize', 14)
    xlabel('Initial Positions','FontSize', 16)
    ylabel('Steps to Terminate','FontSize', 16)
    title("Steps to Terminate",'FontSize', 16)
    exportgraphics(gcf,result_file_directory+"Steps to Terminate.pdf");

    
end


function plot_cpomdp_results(results,M,T,K,e,result_file_directory,method)
    ADF_decision_results=results{2};
    states=results{3};
    measurements=results{4};
    ADF_beliefs=results{5};
    rewards=results{6};
    rewards_collections=results{7};
    X = -25:0.01:25;
    T_temp = length(ADF_decision_results);
    state_dist_at_steps = zeros([T_temp,length(X)]);
    result_means = [];
    result_vars = [];
    for t=1:T_temp
        belief = ADF_beliefs{t};
        if method=="ADF"
            p = cellfun(@(x)(x{1}.*normpdf(X,x{2},sqrt(x{3}))),belief,'UniformOutput',false);
            a = zeros([1,length(X)]);
            for i=1:M
                a = a + p{i};
            end
            %             if T_temp<100
            %                 figure,
            %                 plot(X,a)
            %                 title("Posterior Belief at t="+string(t))
            %                 exportgraphics(gcf,result_file_directory+"Posterior Belief at t="+string(t)+".pdf");
            %                 close all
            %             end
            %
            state_dist_at_steps(t,:)=a';
            ws = cellfun(@(x)(x{1}),belief);
            ms = cellfun(@(x)(x{2}),belief);
            vs = cellfun(@(x)(x{3}),belief);
            [rst_mean,rst_var] = compute_moments_of_gmm({ws',ms',vs'});
            result_means = [result_means;rst_mean];
            result_vars = [result_vars;rst_var];
            %         a = a./M;
            %         figure,
            %         plot(X,a,LineWidth=1)
            %         xlabel("State");
            %         ylabel("Density");
            %         title("Posterior Belief t="+string(t),'FontSize', 14)
            %         hold off
            %         saveas(gcf,result_file_directory+"Posterior Belief t="+string(t)+'.png');
        else
            weights = belief{1};
            samples = belief{2};
            x_indices = round(round(samples,2)*100-X(1)*100+1);
            probs = zeros([1,length(X)]);
            oob_cnt = 0;
            for i=1:length(weights)
                x_index = x_indices(i);
                if x_index>5001 || x_index<1
                    oob_cnt = oob_cnt+1;
                else
                    probs(x_index) = probs(x_index)+weights(i);
                end
                
            end
            disp(oob_cnt)
            state_dist_at_steps(t,:)=probs;
            result_means = [result_means;dot(weights,samples)];
        end
        
    end
    experiment_result = {result_means,result_vars,X,state_dist_at_steps,states,measurements,"C-POMDP"};
    if not(isfolder(result_file_directory+"inferences/"))
        mkdir(result_file_directory+"inferences/")
    end
    if not(isfolder(result_file_directory+"rewards/"))
        mkdir(result_file_directory+"rewards/")
    end
    if not(isfolder(result_file_directory+"motions/"))
        mkdir(result_file_directory+"motions/")
    end
    plot_result(T_temp,experiment_result,"Inference result of C-POMDP,T="+string(T)+",K="+string(K)+",Run="+string(e),"red",result_file_directory+"inferences/","best");
    %% plot the rewards gained
%     figure,
%     bar(1:T_temp,rewards_collections(:,1:3));
%     hold off
%     legend("Left MI","Right MI","Enter MI",'Location','best','FontSize', 12)
%     xlabel('Time','FontSize', 14)
%     ylabel('MI Reward','FontSize', 14)
%     title("MI Rewards Run="+string(e),'FontSize', 14)
%     exportgraphics(gcf,result_file_directory+"rewards/"+"MI Rewards Run="+string(e)+".pdf");
% 
%     figure,
%     bar(1:T_temp,rewards_collections(:,4:6));
%     hold off
%     legend("Left Expected Explicit","Right Expected Explicit","Enter Expected Explicit",'Location','best','FontSize', 12)
%     xlabel('Time','FontSize', 14)
%     ylabel('Reward','FontSize', 14)
%     title("Explicit Rewards Run="+string(e),'FontSize', 14)
%     exportgraphics(gcf,result_file_directory+"rewards/"+"Explicit Rewards Run="+string(e)+".pdf");
% 
%     figure,
%     bar(1:T_temp,rewards_collections(:,7:9));
%     hold off
%     legend("Left Total","Right Total","Enter Total",'Location','best','FontSize', 12)
%     xlabel('Time','FontSize', 14)
%     ylabel('Reward','FontSize', 14)
%     title("Total Rewards Run="+string(e),'FontSize', 14)
%     exportgraphics(gcf,result_file_directory+"rewards/"+"Total Rewards Run="+string(e)+".pdf");
% 
%     %% Plot the robot motion
% 
%     figure,
%     plot(1:T_temp,states)
%     xlabel("Time");
%     ylabel("State");
%     title("Robot motion Run="+string(e),'FontSize', 14)
%     hold off
%     saveas(gcf,result_file_directory+"motions/"+"Robot motion plot Run="+string(e)+".png");
%     close all

end


function plot_initial_belief(initial_belief)
    X=-22:0.05:22;
    Ys = cellfun(@(x)(normpdf(X,x{2},sqrt(x{3}))),initial_belief,"UniformOutput",false);
    Y = zeros([1,length(X)]);
    for j=1:length(Ys)
       Y = Y+1/length(Ys).*Ys{j};
    end
    figure,plot(X,Y,LineWidth=1)
end

function plot_marginal_likelihood(measurement_model)
    gO = measurement_model{2};
    X = 0:0.005:5;
    Ys = cellfun(@(x)(normpdf(X,x{2},sqrt(x{3}))),gO,"UniformOutput",false);
    Y = zeros([1,length(X)]);
    for j=1:length(Ys)
       Y = Y+1/length(Ys).*Ys{j};
    end
    figure,plot(X,Y,LineWidth=1)
end

function plot_state_given_obs(measurement_model,obs_list,X,result_file_directory)
    gS = measurement_model{1};
    gO = measurement_model{2};
    n_of_obs = length(obs_list);
    legend_list = {};
    min_obs = min(obs_list);
    max_obs = max(obs_list);
    line_styles = [":","-.","-","--"];
    figure,
    for i=1:n_of_obs
        obs = obs_list(i);
        weights = cellfun(@(x)(normpdf(obs,x{2},sqrt(x{3}))),gO);
        weights = weights./sum(weights);
        Ys = cellfun(@(x)(normpdf(X,x{2},sqrt(x{3}))),gS,"UniformOutput",false);
        Y = zeros([1,length(X)]);
        for j=1:length(Ys)
           Y = Y+weights(j).*Ys{j};
        end
        plot(X,Y,LineWidth=1,LineStyle=line_styles(i))
%         legend_list{end+1}="measurement="+string(obs);
        hold on
    end
    hold off
    legend("Left edge","Right edge","Corridor","Door")
    xlabel("Position")
    ylabel("Density")
    title("State Distribution for different features")
    exportgraphics(gcf,result_file_directory+"State Distribution different features.pdf");
    
end

function plot_obs_given_state(measurement_model,state,result_file_directory)
    gS = measurement_model{1};
    gO = measurement_model{2};
    X = 0:0.005:5;
    weights = cellfun(@(x)(normpdf(state,x{2},sqrt(x{3}))),gS);
    weights = weights./sum(weights);
    Ys = cellfun(@(x)(normpdf(X,x{2},sqrt(x{3}))),gO,"UniformOutput",false);
    Y = zeros([1,length(X)]);
    for j=1:length(Ys)
       Y = Y+weights(j).*Ys{j};
    end
    figure,plot(X,Y,LineWidth=1)
    title("Obs Distribution at State"+string(state)+'.png')
    saveas(gcf,result_file_directory+"Obs Distribution at State"+string(state)+'.png');
end


function plot_obs_model(measurement_model,result_file_directory)
    gS = measurement_model{1};
    gO = measurement_model{2};
    s=-21:0.25:21;
    o=0:0.25:5;
    ns=length(s);
    no=length(o);
    sp=zeros(ns,no);
    for i=1:ns
      for j=1:no
        sp(i,j)=sum(1/22.*cellfun(@(g)(normpdf(s(i),g{2},sqrt(g{3}))),gS).*cellfun(@(g)(normpdf(o(j),g{2},sqrt(g{3}))),gO));
        
      end
    end
    figure,
    surf(s,o,sp');
    xlabel("State");
    ylabel("Measurment");
    title("Joint Distribution",'FontSize', 14)
    hold off
    exportgraphics(gcf,result_file_directory+"Measurement model"+'.pdf');
end


function plot_reward_model(explicit_reward_model,result_file_directory)
    X=-21:0.25:21;
    figure,
    for i=1:length(explicit_reward_model)
        reward_func = explicit_reward_model{i};
        Ys = cellfun(@(x)(x{1}.*normpdf(X,x{2},sqrt(x{3}))),reward_func,"UniformOutput",false);
        Y = zeros([1,length(X)]);
        for j=1:length(Ys)
            Y = Y+Ys{j};
        end
        plot(X,Y,LineWidth=1),
        hold on
    end
    legend("Left","Right","Enter");
    hold off
    xlabel("States");
    ylabel("Reward");
    title("Reward Models",'FontSize', 14)
    exportgraphics(gcf,result_file_directory+"Reward Model"+'.pdf');
end


function [step_results,ls_reward_results,total_rewards_results,last_distance_results]=calculate_performance_stats(result_collections,initial_states,E)
    n_of_collections = length(result_collections);
    gate_position = 3;
    median_steps_per_start = zeros([length(initial_states),n_of_collections]);
    steps_std_per_start = zeros([length(initial_states),n_of_collections]);
    mean_steps_per_start = zeros([length(initial_states),n_of_collections]);
    worst_steps_per_start = zeros([length(initial_states),n_of_collections]);
    best_steps_per_start = zeros([length(initial_states),n_of_collections]);

    median_final_distance_per_start = zeros([length(initial_states),n_of_collections]);
    final_distance_std_per_start = zeros([length(initial_states),n_of_collections]);
    worst_final_distance_per_start = zeros([length(initial_states),n_of_collections]);
    best_final_distance_per_start = zeros([length(initial_states),n_of_collections]);
    mean_final_distance_per_start = zeros([length(initial_states),n_of_collections]);

    median_final_reward_per_start = zeros([length(initial_states),n_of_collections]);
    final_reward_std_per_start = zeros([length(initial_states),n_of_collections]);
    worst_final_reward_per_start = zeros([length(initial_states),n_of_collections]);
    best_final_reward_per_start = zeros([length(initial_states),n_of_collections]);
    mean_final_reward_per_start = zeros([length(initial_states),n_of_collections]);

    median_total_reward_per_start = zeros([length(initial_states),n_of_collections]);
    total_reward_std_per_start = zeros([length(initial_states),n_of_collections]);
    worst_total_reward_per_start = zeros([length(initial_states),n_of_collections]);
    best_total_reward_per_start = zeros([length(initial_states),n_of_collections]);
    mean_total_reward_per_start = zeros([length(initial_states),n_of_collections]);

    for i=1:length(initial_states)
        for k=1:n_of_collections
            steps = zeros([E,1]);
            explicit_rewards = zeros([E,1]);
            total_rewards = zeros([E,1]);
            last_distance = zeros([E,1]);
            results_collection = result_collections{k};
            for j=1:E
                results = results_collection{i,j};
                explicit_rewards(j)= results{6}(end,3);
                steps(j)= length(results{2});
                last_distance(j)=abs(results{2}(end)-gate_position);
                total_rewards(j)= sum(results{6}(:,3));
            end
            median_steps_per_start(i,k)=median(steps);
            mean_steps_per_start(i,k)=mean(steps);
            worst_steps_per_start(i,k)=max(steps);
            best_steps_per_start(i,k)=min(steps);
            steps_std_per_start(i,k)=std(steps);

            median_final_reward_per_start(i,k)=median(explicit_rewards);
            mean_final_reward_per_start(i,k)=mean(explicit_rewards);
            worst_final_reward_per_start(i,k)=min(explicit_rewards);
            best_final_reward_per_start(i,k)=max(explicit_rewards);
            final_reward_std_per_start(i,k)=std(explicit_rewards);

            median_final_distance_per_start(i,k)=median(last_distance);
            mean_final_distance_per_start(i,k)=mean(last_distance);
            worst_final_distance_per_start(i,k)=max(last_distance);
            best_final_distance_per_start(i,k)=min(last_distance);
            final_distance_std_per_start(i,k)=std(last_distance);

            median_total_reward_per_start(i,k)=median(total_rewards);
            mean_total_reward_per_start(i,k)=mean(total_rewards);
            worst_total_reward_per_start(i,k)=min(total_rewards);
            best_total_reward_per_start(i,k)=max(total_rewards);
            total_reward_std_per_start(i,k)=std(total_rewards);
        end

    end
    step_results = {median_steps_per_start,mean_steps_per_start,worst_steps_per_start,best_steps_per_start,steps_std_per_start};
    ls_reward_results = {median_final_reward_per_start,mean_final_reward_per_start,worst_final_reward_per_start,best_final_reward_per_start,final_reward_std_per_start};
    total_rewards_results ={median_total_reward_per_start,mean_total_reward_per_start,worst_total_reward_per_start,best_total_reward_per_start,total_reward_std_per_start};
    last_distance_results = {median_final_distance_per_start,mean_final_distance_per_start,worst_final_distance_per_start,best_final_distance_per_start,final_distance_std_per_start};
end