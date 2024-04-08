function multid_tests()
    %% test mutivariate model generator
    rnd_seed = 2023;
     D = 5;
    prior_mu = 0;
    prior_sigma = 1;
    num_of_comp = 2;
    dim = 2;
    transition_models = generate_transition_model(D,prior_mu,prior_sigma,num_of_comp,dim,rnd_seed);
    model_d1 = transition_models{1};
    assert(sum(model_d1{1})==1)
    assert(det(model_d1{4}(:,:,1))>=0)
    measurement_model = generate_measurement_model(dim,rnd_seed);
    assert(det(measurement_model{3})>=0)
    %% test range generator
     K = 5;
    initial_state = [0,1];
    T = 20;
    initial_mean = [0,0]; 
    initial_cov = [1,0;0,1];
    initial_model = {initial_mean,initial_cov};
    state_dim = 2;
    measurement_dim = 2;

    convergence_threshold=0.0001;
    fixed_settings = {};
    
    %% test results
    
    [EP_results,ADF_results,random_results,pf_results,ADF_simple_results,ADF_gmm_unc_results]= clgsdm_GMM_discrete_multi(initial_state,initial_model,transition_models,measurement_model,T,K,state_dim,measurement_dim,convergence_threshold,fixed_settings,rnd_seed);

    EP_states = EP_results{2};
    ADF_simple_states = ADF_simple_results{2};
    PF_states = pf_results{2};

    EP_measurements = EP_results{3};
    ADF_simple_measurements = ADF_simple_results{3};
    PF_measurements = pf_results{3};

    EP_means = EP_results{4};
    ADF_simple_means = ADF_simple_results{4};
%     TODO: change the order of PF results to align with others
    PF_means = pf_results{5};

%     EP_min = min(min(EP_states),min(EP_measurements));
    EP_max = max(max(EP_states),max(EP_measurements));

%     ADF_simple_min = min(min(ADF_simple_states),min(ADF_simple_measurements));
    ADF_simple_max = max(max(ADF_simple_states),max(ADF_simple_measurements));

%     PF_min = min(min(PF_states),min(PF_measurements));
    PF_max = max(max(PF_states),max(PF_measurements));

    X_state_measurement_max = max(max(EP_max,ADF_simple_max),PF_max);

    EP_mean_max = max(EP_means);
    ADF_mean_max = max(ADF_simple_means);
    PF_mean_max = max(PF_means);
    estimated_mean_max = max(max(EP_mean_max,ADF_mean_max),PF_mean_max);

    X_max = max(estimated_mean_max,X_state_measurement_max);
    
    saved_location = "experiment_results/greedy ep/online decisions/2D_results/";

    draw_2d_inference(EP_states,EP_measurements,EP_means,"EP",saved_location,T);
    draw_2d_inference(ADF_simple_states,ADF_simple_measurements,ADF_simple_means,"ADF",saved_location,T);
    draw_2d_inference(PF_states,PF_measurements,PF_means,"PF",saved_location,T);

    EP_state_inference_error = zeros([T,1]);
    ADF_simple_state_inference_error = zeros([T,1]);
    PF_state_inference_error = zeros([T,1]);
    for t=1:T
        EP_state_inference_error(t,:) = norm(EP_means(t,:)-EP_states(t,:));
        ADF_simple_state_inference_error(t,:) = norm(ADF_simple_means(t,:)-ADF_simple_states(t,:));
        PF_state_inference_error(t,:) = norm(PF_means(t,:)-PF_states(t,:));
    end

    figure,
    plot(1:T,EP_state_inference_error,"LineWidth",1)
    hold on
    plot(1:T,ADF_simple_state_inference_error,"LineWidth",1)
    hold on
    plot(1:T,PF_state_inference_error,"LineWidth",1)
    legend("EP","ADF","PF")
    xlabel('Timesteps','FontSize', 14)
    ylabel('Estimation Error','FontSize', 14)
    title("Estimation Error",'FontSize', 14)
    hold off
    saveas(gcf,saved_location+"Estimation_error.pdf");
end

function draw_2d_inference(states,measurements,estimated_means,method_name,saved_location,T)
    figure,
    plot(states(:,1),states(:,2),"->","LineWidth",1,"MarkerSize",5)
    hold on
    text(states(:,1),states(:,2), string(1:T), 'FontSize',12,'Interpreter','latex')
    hold on
    plot(measurements(:,1),measurements(:,2),"->","LineWidth",1,"MarkerSize",5)
    hold on
    text(measurements(:,1),measurements(:,2), string(1:T), 'FontSize',12,'Interpreter','latex')
    hold on
    plot(estimated_means(:,1),estimated_means(:,2),"->","LineWidth",1,"MarkerSize",5)
    hold on
    text(estimated_means(:,1),estimated_means(:,2), string(1:T), 'FontSize',12,'Interpreter','latex')
    hold on
    legend(method_name+" states",method_name+" measurements",method_name+" estimated means")
    xlabel('X1','FontSize', 14)
    ylabel('X2','FontSize', 14)
    title(method_name+" inference results",'FontSize', 14)
    hold off
    saveas(gcf,saved_location+method_name+".pdf");
end
