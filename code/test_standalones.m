function test_standalones()
%% MI global bound
    H=1;
    b=1;
    R=2;
    mu_0 = 1;
    var_0 = 1.6;
    T = 5;
    num_of_comp = 2;
    transition_models = generate_transition_model(3,1,2,num_of_comp,1,"");
    dynamic_1= transition_models{1};
    true_beliefs = {};
    % ADF approximation
    approximated_beliefs = {};

    initial_belief = set_init_belief({},mu_0,var_0);
    weights = dynamic_1{1};
    coefs = dynamic_1{2};
    means = dynamic_1{3};
    vars = dynamic_1{4};
    initial_state = 0;
    states = zeros([T,1]);
    measurements = zeros([T,1]);
    measurement_model =  {H,b,R};
    for t=1:T
        true_belief = initial_belief;
        approximated_belief = initial_belief;
        last_state = initial_state;
        if t~=1
            true_belief = true_beliefs{t};
            approximated_belief = approximated_beliefs{t};
            last_state = states(t-1);
        end
        true_augmented_dist = {};
        true_augmented_dist.num_of_comp = true_belief.num_of_comp*num_of_comp;   
        true_augmented_dist.weights = zeros([true_augmented_dist.num_of_comp,1]);
        true_augmented_dist.mus = zeros([true_augmented_dist.num_of_comp,2]);
        true_augmented_dist.vars = zeros([true_augmented_dist.num_of_comp,2,2]);
        % compute augmented distribution of the truth
        for j = 1:true_belief.num_of_comp
            true_joint_model=directly_compute_moment_matching(dynamic_1,measurement_model,true_belief.mus(j),true_belief.vars(j));
            indices = ((j-1)*num_of_comp+1):(j*num_of_comp);
            true_augmented_dist.weights(indices)=true_joint_model{1}.*true_belief.weights(j);
            true_augmented_dist.mus(indices,:)=true_joint_model{2};
            true_augmented_dist.vars(indices,:,:)=true_joint_model{3};
        end

        [gloabl_approx_mean,gloabl_approx_cov]=compute_moments_of_gmm({true_augmented_dist.weights,true_augmented_dist.mus,true_augmented_dist.vars});

        % compute the augmented distribution of the approximation
        approx_joint_model=directly_compute_moment_matching(dynamic_1,measurement_model,approximated_belief.mus,approximated_belief.vars);
        [approx_joint_mean,approx_joint_cov]=compute_moments_of_gmm(approx_joint_model);
        
        % test point 1, if these two moments are the same.
        
%         simulation for next step
        [next_state] = dynamic_transition(last_state,1,transition_models);
         next_measurement = measure(next_state,measurement_model);
         states(t) = next_state;
         measurements(t) = next_measurement;
        
%          posterior update
        [forward_mean_cov_new]=update_after_measurement(dynamic_1,measurement_model,next_measurement,{approximated_belief.mus,approximated_belief.vars});
        new_belief = approximated_belief;
        new_belief.mus = forward_mean_cov_new{1};
        new_belief.vars = forward_mean_cov_new{2};
        approximated_beliefs{end}=new_belief;
    end
end


function initial_belief=set_init_belief(initial_belief,mu,variance)
    initial_belief.num_of_comp = 1;
    initial_belief.weights = 1;
    initial_belief.mus = mu;
    initial_belief.vars = variance;
end