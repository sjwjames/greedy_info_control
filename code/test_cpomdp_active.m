function test_cpomdp_active()
%% Environment setup
    debug_flag = false;
    T = 50;
    K = 3;
    num_of_comp = 1;
    result_file_directory = "experiment_results/greedy ep/online decisions/cpomdp_active/n="+string(num_of_comp)+",k="+string(K)+"/";
%     initial_state = 20;
    agent_transition_models = generate_transition_model(K,1,2,num_of_comp,1,"CPOMDP");
    if not(isfolder(result_file_directory))
        mkdir(result_file_directory)
    end
    
    so=1.6;
    means = -21:2:21;
    M=length(means);
    initial_belief = {ones([M,1])/M,means',ones([M,1])*so};
    target_transition_model = {[0.4;0.4;0.2],[1;1;1],[2;-2;0],[0.05;0.05;0.05]};
    
    % plot_initial_belief(initial_belief);

%% Run C-POMDP active information acquisition experiment with 1-shot ADF greedy information control

    initial_states = 10;
    E = 11;
    run_flags = [true,true];
    if run_flags(1)
        results_collection = cell(length(initial_states),E);
    end
    
    if run_flags(2)
        PF_results_collection = cell(length(initial_states),E);
    end

    initial_agent_state = 0;
    
    N= 1000;
    verbose = true;
    
    debug_flag = false;
    ADF_agent_state_collections = {};
    ADF_target_state_collections = {};
    ADF_measurement_collections = {};
    ADF_times = zeros([length(initial_states),E]);
    PF_agent_state_collections = {};
    PF_target_state_collections = {};
    PF_measurement_collections = {};
    PF_times = zeros([length(initial_states),E]);
%     f(length(initial_states),E,4) = parallel.FevalFuture;
    for i = 1:length(initial_states)
        initial_target_state = initial_states(i);
        for e = 1:E
            %         initial_state = randi([-20,20],1);
%             f(i,e,:)=parfeval(@info_cpomdp,4,transition_models,measurement_model,initial_belief,initial_state,M,N,explicit_reward_model,T,[-25,25],{},true);
%             [ADF_results,PF_results,ADF_ex_results,PF_ex_results] = fetchOutputs(f(i,e,:));
            [ADF_results,PF_results]=info_cpomdp_active(agent_transition_models,target_transition_model,initial_belief,initial_target_state,initial_agent_state,N,T,[-23,23],{},verbose,run_flags);           
%           
            if run_flags(1)
                results_collection{i,e}=ADF_results;
                ADF_agent_state_collections{end+1}= ADF_results{3};
                ADF_target_state_collections{end+1}= ADF_results{4};
                ADF_measurement_collections{end+1}= ADF_results{5};
                ADF_times(i,e) = mean(ADF_results{end});
            end
   
            
            if run_flags(2)
                PF_results_collection{i,e}=PF_results;
                PF_agent_state_collections{end+1}= PF_results{3};
                PF_target_state_collections{end+1}= PF_results{4};
                PF_measurement_collections{end+1}= PF_results{5};
                PF_times(i,e) = mean(PF_results{end});
            end

        end
        close all
    end


  %% plot movement of the agent and target
    close all    
    
    % background = imread("hall.png");
    % targetimg = imread("target.png");
    % agentimg = imread("agent.png");
    % background = imresize(background, 0.5);
    % agentimg = imresize(agentimg, size(targetimg,2)/size(agentimg,2));
    % scale_factor = size(targetimg,2)*42/size(background,2);
    % background = imresize(background, size(targetimg,2)*42/size(background,2));

    
    figh = figure;
    
    h1 = plot(initial_agent_state,5,'ro', 'MarkerSize', 10,'LineWidth',3);
    hold on
    h2 = plot(initial_states,5,'bo', 'MarkerSize', 10,'LineWidth',3);
    hold on
    legend("Agent","Target")
    clear movie_frames
    for i = 1:T
        % current_agent_pos = initial_agent_state;
        % current_target_pos = initial_states;
        % if i~=1
        %     current_agent_pos = ADF_results{4}(i-1);
        %     current_target_pos = ADF_results{3}(i-1);
        % end
        xlim([-25 25])
        ylim([0 11])
        current_agent_pos = ADF_results{3}(i);
        current_target_pos = ADF_results{4}(i);
        % plot(current_agent_pos,5,'ro', 'MarkerSize', 10,'LineWidth',3)
        % hold on
        % plot(current_target_pos,5,'bo', 'MarkerSize', 10,'LineWidth',3)
        % hold on
        set(h1,'xdata',current_agent_pos,'ydata',6) ;
        set(h2,'xdata',current_target_pos,'ydata',6) ;
        title("Agent and target at t="+string(i))
        grid on
        movie_frames(i)=getframe(figh,[10 10 520 400]);
        
    end
    hold off
    mywriter = VideoWriter(result_file_directory+"active_info","MPEG-4");
    mywriter.FrameRate = 10;
    open(mywriter);
    writeVideo(mywriter,movie_frames);
    close(mywriter);
    

    close all
    figh = figure;
    
    h1 = plot(initial_agent_state,5,'ro', 'MarkerSize', 10,'LineWidth',3);
    hold on
    h2 = plot(initial_states,5,'bo', 'MarkerSize', 10,'LineWidth',3);
    hold on
    legend("Agent","Target")
    clear movie_frames
    for i = 1:T
        % current_agent_pos = initial_agent_state;
        % current_target_pos = initial_states;
        % if i~=1
        %     current_agent_pos = ADF_results{4}(i-1);
        %     current_target_pos = ADF_results{3}(i-1);
        % end
        xlim([-25 25])
        ylim([0 11])
        current_target_pos = PF_results{4}(i);
        current_agent_pos = PF_results{3}(i);
        % plot(current_agent_pos,5,'ro', 'MarkerSize', 10,'LineWidth',3)
        % hold on
        % plot(current_target_pos,5,'bo', 'MarkerSize', 10,'LineWidth',3)
        % hold on
        set(h1,'xdata',current_agent_pos,'ydata',6) ;
        set(h2,'xdata',current_target_pos,'ydata',6) ;
        title("Agent and target at t="+string(i))
        grid on
        movie_frames(i)=getframe(figh,[10 10 520 400]);
        
    end
    hold off
    mywriter = VideoWriter(result_file_directory+"active_info_PF","MPEG-4");
    mywriter.FrameRate = 10;
    open(mywriter);
    writeVideo(mywriter,movie_frames);
    close(mywriter);
 %% load data and calculate stats
    ADF_color = 'r';
    pf_color = 'y';
    ADF_agent_pos = ADF_results{3};
    ADF_agent_pos = [initial_agent_state;ADF_agent_pos];
    target_pos = ADF_results{4};
    target_pos = [initial_states;target_pos];
    PF_agent_pos = PF_results{3};
    PF_agent_pos = [initial_agent_state;PF_agent_pos];
    ADF_measurements = ADF_results{5};
    PF_measurements = PF_results{5};
    figure,
    plot(0:T,ADF_agent_pos,"Color",ADF_color,"LineWidth",3)
    hold on
    plot(0:T,PF_agent_pos,"Color",pf_color,"LineWidth",3)
    hold on
    plot(0:T,target_pos,"Color","b","LineWidth",3)
    hold on
    scatter(1:T,ADF_measurements,"*","Color",ADF_color)  
    hold on
    scatter(1:T,PF_measurements,"*","Color",pf_color)
    hold off
    legend("Agent Position with ADF","Agent Position with PF","Target Position","Measurement with ADF","Measurement with PF",'FontSize', 12,'Location','best')
    xlabel("Time",'FontSize', 14)
    ylabel("Position or Measurement",'FontSize', 14)
    title("Movement of Agent and Target",'FontSize', 14)
    exportgraphics(gcf,result_file_directory+"movement.pdf");    

  %% calculate cumulative MI.
    
    ADF_MIs=calculate_cumulative_MI(ADF_agent_state_collections,ADF_measurement_collections,target_transition_model);

    PF_MIs=calculate_cumulative_MI(PF_agent_state_collections,PF_measurement_collections,target_transition_model);
  %% plot MI.

    ADF_MI_mean = mean(ADF_MIs,1);
    PF_MI_mean = mean(PF_MIs,1);
    
    ADF_MI_std = std(ADF_MIs);
    PF_MI_std = std(PF_MIs);
    X=1:T;
    X2 = [X,fliplr(X)];
    ADF_inbetween = [ADF_MI_mean+ADF_MI_std,fliplr(ADF_MI_mean-ADF_MI_std)];
    pf_inbetween = [PF_MI_mean+PF_MI_std,fliplr(PF_MI_mean-PF_MI_std)];
    
    figure,
    
    f4=fill(X2,pf_inbetween, 'black');
    hold on
    f6 = fill(X2,ADF_inbetween,'m');
    hold on
    
    plot(X,PF_MI_mean,'-x','Color','black','LineWidth',3)
    hold on
    
    plot(X,ADF_MI_mean,'-x','Color','m','LineWidth',3)
    hold on
    
    alpha(f4,0.2)
    alpha(f6,0.2)
    legend('','','PF','ADF','Location','southwest','FontSize', 12)
%     legend('EP','ADF-GMM','Random','PF','ADF-GMM unconstrained','ADF-Gaussian','Location','southwest','FontSize', 12)
    xlabel('Timesteps','FontSize', 14)
    ylabel('Realized MI','FontSize', 14)
    title("Cumulative MI Evaluated by Numerical Approximation",'FontSize', 14)
    hold off
    exportgraphics(gcf,result_file_directory+'Cumulative MI Evaluated by Numerical Approximation.pdf');    

 %% Runtime plot
    ADF_time_collection = [];
    PF_time_collection = [];
    for i=1:length(initial_states)
        for e=1:E
            ADF_time_collection = [ADF_time_collection;results_collection{i,e}{end}];
            PF_time_collection = [PF_time_collection;PF_results_collection{i,e}{end}];
        end
    end
    ADF_time_mean = mean(ADF_time_collection,1);
    PF_time_mean = mean(PF_time_collection,1);
    ADF_time_std  = std(ADF_time_collection,1);
    PF_time_std  = std(PF_time_collection,1);
    figure,
    errorbar(1:T,ADF_time_mean,ADF_time_std,ADF_time_std,"Linewidth",3)
    hold on
    errorbar(1:T,PF_time_mean,PF_time_std,PF_time_std,"Linewidth",3)
    legend("ADF","PF",'Location','southwest','FontSize', 12)
    xlabel('Timesteps','FontSize', 14)
    ylabel('Decision Time','FontSize', 14)
    title("Average Decision Time Per Timestep",'FontSize', 14)
    exportgraphics(gcf,result_file_directory+'Average Runtime.pdf');  
 %% Distance plot
    ADF_distances = [];
    PF_distances = [];
    for i =1:length(ADF_target_state_collections)
        ADF_distances = [ADF_distances;abs(ADF_agent_state_collections{i}-ADF_target_state_collections{i})'];
        PF_distances = [PF_distances;abs(PF_agent_state_collections{i}-PF_target_state_collections{i})'];
    end
    ADF_distance_mean = mean(ADF_distances,1);
    PF_distance_mean = mean(PF_distances,1);
    ADF_distance_std  = std(ADF_distances,1);
    PF_distance_std  = std(PF_distances,1);
    figure,
    errorbar(1:T,ADF_distance_mean,ADF_distance_std,ADF_distance_std,"Linewidth",3)
    hold on
    errorbar(1:T,PF_distance_mean,PF_distance_std,PF_distance_std,"Linewidth",3)
    legend("ADF","PF",'Location','southwest','FontSize', 12)
    xlabel('Timesteps','FontSize', 14)
    ylabel('Average Distance','FontSize', 14)
    title("Average Distance between Agent and Target",'FontSize', 14)
    exportgraphics(gcf,result_file_directory+'Average Distance between Agent and Target.pdf');  
end


function MI=calculate_cumulative_MI(agent_states_collections,measurement_collections,target_transition_model)
    Z = linspace(-25,25,3000)';
    delta_state = Z(2)-Z(1);
    N = length(Z);
    E = length(agent_states_collections);
    T = length(agent_states_collections{1});
    MI = zeros([E,T]);
    weights_vector = target_transition_model{1};
    coefficients_vector = target_transition_model{2};
    mu_vector = target_transition_model{3};
    var_vector = target_transition_model{4};
    discretized_target_transition_probs = [];
    rep_Zt1 = repmat(Z,[N,1]);
    rep_Zt = repelem(Z,N);
    for i = 1:length(weights_vector)
        d = weights_vector(i).*normpdf(rep_Zt1,rep_Zt.*coefficients_vector(i)+mu_vector(i),sqrt(var_vector(i)));
        discretized_target_transition_probs = [discretized_target_transition_probs;d'];
    end
    discretized_target_transition_probs = reshape(sum(discretized_target_transition_probs,1),[N,N])';
    for e = 1:E
        current_belief_probs = ones([N,1])./N;
        agent_states = agent_states_collections{e};
        measurements = measurement_collections{e};
        for t=1:T
            x = agent_states(t);
            y = measurements(t);
            % p(Z_t)
            marg_target_probs = (sum(discretized_target_transition_probs.*current_belief_probs,1).*delta_state)';
            % p(y_t|Z_t,x_t)
            moments = get_measurement_moments(Z,x);
            cond_ll_probs = normpdf(y,moments{1},sqrt(moments{2})); 
            % p(Z_t,y_t|x_t)
            joint_probs = cond_ll_probs.*marg_target_probs;
            % p(Z_t|y_t,x_t)
            post_probs = joint_probs./sum(joint_probs);
            nonz_marg_target_probs = marg_target_probs(marg_target_probs~=0);
            nonz_post_probs = post_probs(post_probs~=0);
            MI(e,t) = dot(nonz_marg_target_probs,-log(nonz_marg_target_probs))*delta_state-dot(nonz_post_probs,-log(nonz_post_probs))*delta_state;
            current_belief_probs = post_probs;
        end
    end
end




