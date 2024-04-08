function [q_table,decision_collections,states_collections,measurement_collections,mc_bf_collections]=mc_closed_optimal(N_particles,T,D,K,N_beliefs,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num,input_q_table,state_range,file_direc)
    q_table = input_q_table;
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    states_collections = [];
    measurement_collections = [];
    mc_bf_collections = {};
    decision_collections = [];
    effective_sample_threshold = 100;
    greedy_flag = false;
    debug_flag = false;
    num_of_comp = length(transition_models{1}{1});
    q_table_change_flag = true;
    while epoch_num>0 && q_table_change_flag
        init_samples=normrnd(initial_model{1},initial_model{2},[N_particles,1]);
        init_weights = ones([N_particles,1]).*(1/N_particles);
        blief_states = {{init_weights,init_samples}};
        states = [];
        measurements = [];
        decisions = [];
        out_of_bound_flag = false;
        q_table_change_flag= false;
        for t=1:T
            current_belief_state = blief_states{t};
            if ~greedy_flag
                k_nearest_result = find_nearest_neighbours(q_table,t,K,current_belief_state,distance_threshold);
            else
                k_nearest_result = {};
            end
           
            current_q_vals = [];
%             No enough nearest neighbours that satisfy the distance
%             condition
            if isempty(k_nearest_result)
                    current_q_vals = estimate_q_vals_by_sampling(D,current_belief_state,transition_models,measurement_model,N_beliefs,greedy_flag,N_particles,num_of_comp,K,distance_threshold,q_table,t);
                if length(q_table)<t
                    q_table{t} = {{blief_states{t},current_q_vals}};
                else
                    q_table{t}{end+1}={blief_states{t},current_q_vals};
                end
                    q_table_change_flag = q_table_change_flag||true;
            else
               
                if debug_flag
                    current_q_vals = estimate_q_vals_by_sampling(D,current_belief_state,transition_models,measurement_model,N_beliefs,greedy_flag,N_particles,num_of_comp,K,distance_threshold,q_table,t);
                    q_table_change_flag = q_table_change_flag||true;
%                     figure,
%                     bar(1:D,[q_vals;sampled_q_vals])
%                     legend("KNN","Sampling",'Location','best','FontSize', 12)
%                     xlabel('Control Decision','FontSize', 14)
%                     ylabel('Q values','FontSize', 14)
%                     title("Q values estimated by sampling v.s. KNN",'FontSize', 14)
%                     exportgraphics(gcf,file_direc+"Q values estimated by sampling v.s. KNN,t="+string(t)+".pdf");
                else
                    k_distances = k_nearest_result{1};
                    k_nearest_q_vals = k_nearest_result{3};
                    k_indices = k_nearest_result{4};
                    distance_inverse = 1./k_distances;
                    q_vals = [];
                    for d = 1:D
                        q_val = (distance_inverse*k_nearest_q_vals(:,d))/sum(distance_inverse);
                        if q_val<0
                            disp(distance_inverse)
                            disp(k_nearest_q_vals(:,d))
                        end
                        q_vals = [q_vals,q_val];
                    end
                    current_q_vals = q_vals;
                    q_table_change_flag = q_table_change_flag||false;
                end
            end
            

            [max_val,decision_to_make] = max(current_q_vals);
            decisions = [decisions,decision_to_make];
%             simulation
            last_state = initial_state;
            if t~=1
               last_state = states(t-1);
            end
            next_state = dynamic_transition(last_state,decision_to_make,transition_models);
            if next_state<=state_range(1)||next_state>=state_range(2)
                out_of_bound_flag = true;
                break;
            end
            states=[states;next_state];
            measurement = measure(next_state,measurement_model);
            measurements=[measurements;measurement];
%             simulation ends, belief state update
            blief_states{end+1} = sample_bs(current_belief_state,decision_to_make,measurement,transition_models,measurement_model,N_particles,effective_sample_threshold,num_of_comp);
        end
        if ~out_of_bound_flag
            states_collections = [states_collections,states];
            measurement_collections = [measurement_collections,measurements];
            decision_collections = [decision_collections;decisions];
            mc_bf_collections{end+1} = blief_states;
            epoch_num = epoch_num-1;
            disp("epoch left:"+string(epoch_num));
            disp(string(decisions));
        end
        
    end
    
end

function sampled_belief_states = forward_sampling(current_belief_state,decision_to_make,sampled_measurements,transition_models,measurement_model,N)
    current_sample_states_collections = [];
    current_sampled_states  = current_belief_state{2};
    current_sampled_weights = current_belief_state{1};
    N_particles = length(current_sampled_weights);
    for i=1:N
        current_sampled_cnts = mnrnd(N_particles,current_sampled_weights);
        current_sampled_states  = repelem(current_sampled_states,current_sampled_cnts);
        current_sample_states_collections = [current_sample_states_collections;current_sampled_states];
    end
    H = measurement_model{1};
    b = measurement_model{2};
    R = measurement_model{3};
    sampled_next_states = dynamic_transition(current_sample_states_collections,decision_to_make,transition_models);
    sampled_next_weights = log_normpdf(repelem(sampled_measurements,N),H*sampled_next_states+b,sqrt(R));
    
    sampled_belief_states = {};
    for i=1:N
        indices = (i-1)*N_particles+1:i*N_particles;
        next_states = sampled_next_states(indices);
        log_next_weights = sampled_next_weights(indices);
        log_next_weights = log_next_weights-max(log_next_weights);
        next_weights = exp(log_next_weights)./sum(exp(log_next_weights));
        sampled_belief_states{end+1} = {next_weights,next_states};
    end
    
end


function sampled_belief_state = sample_bs(current_belief_state,decision_to_make,sampled_measurement,transition_models,measurement_model,N,effective_sample_threshold,num_of_comp)
    

    sampled_states  = current_belief_state{2};
    sampled_weights = current_belief_state{1};
    [samples,weights_new]=reweight_particles(N,num_of_comp,sampled_weights,sampled_states,transition_models{decision_to_make},measurement_model,sampled_measurement,effective_sample_threshold);
   
%     H = measurement_model{1};
%     b = measurement_model{2};
%     R = measurement_model{3};
%     sampled_next_states = dynamic_transition(sampled_states,decision_to_make,transition_models);
%     sampled_next_weights = log_normpdf(sampled_measurement,H*sampled_next_states+b,sqrt(R));
%     
%     sampled_next_weights = log(sampled_weights)+ sampled_next_weights;
%     sampled_next_weights = sampled_next_weights-max(sampled_next_weights);
%     sampled_next_weights = exp(sampled_next_weights)./sum(exp(sampled_next_weights));
%     
%     sampled_belief_state = {sampled_next_weights,sampled_next_states};
%     sample_efficiency = 1/sum(sampled_next_weights.^2);
%     if sample_efficiency<effective_sample_threshold
%          sample_copies = mnrnd(N,sampled_next_weights);
%          weights_new = ones([N,1]).*(1/N);
%          sampled_next_states = repelem(sampled_next_states,sample_copies);
%          sampled_belief_state = {weights_new,sampled_next_states};
%     end

    sampled_belief_state = {weights_new',samples(:,1)};
end


function current_q_vals=estimate_q_vals_by_sampling(D,current_belief_state,transition_models,measurement_model,N_beliefs,greedy_flag,N_particles,num_of_comp,K,distance_threshold,q_table,t)
            current_q_vals=[];
                for d =1:D
                    bs_list = {};
%                     forward sampling from last belief state
                    sampled_next_states = dynamic_transition(current_belief_state{2},d,transition_models);
                    sampled_next_measurements = measure(sampled_next_states,measurement_model);
                    samples = [sampled_next_states,sampled_next_measurements];
%                     measurement_probs = normpdf(sampled_next_measurements,H*sampled_next_states+b,sqrt(R));
%                     generate belief states of the next step from samples
                    if ~greedy_flag
                        bs_list = forward_sampling(current_belief_state,d,sampled_next_measurements,transition_models,measurement_model,N_beliefs);
                    end
                    
% %                     compute marginal likelihoods
%                     measurement_dist_means = [];
%                     sampled_measurements_rep = [];
%                     for i=1:N_particles
%                         sampled_measurements_rep = [sampled_measurements_rep;ones([N_particles-1,1]).*sampled_next_measurements(i)];
%                         measurement_dist_means = [measurement_dist_means;sampled_next_states(1:end~=i).*H+b];
%                     end
%                     marginal_lhs = normpdf(sampled_measurements_rep,measurement_dist_means,sqrt(R));
%                     marginal_lhs = reshape(marginal_lhs,[N_particles-1,N_particles]);
%                     marginal_lhs = sum(marginal_lhs,1)./(N_particles-1);
% %                     compute instatnaneous reward, algorithm line 26
%                     in_reward = 1/N_particles*(sum(log(measurement_probs(measurement_probs~=0)))-sum(log(marginal_lhs(marginal_lhs~=0))));
                    in_reward=particle_MI_estimation(N_particles,num_of_comp,samples,current_belief_state{1},transition_models{d},measurement_model,current_belief_state{2});
                    future_reward = 0;
%                     look for nearest neighbours that satisfy the distance
%                     condition to compute the future rewards.
                    if ~greedy_flag
                        for n=1:N_beliefs
                            bs = bs_list{n};
                            k_nearest_result = find_nearest_neighbours(q_table,t+1,K,bs,distance_threshold);
                            if ~isempty(k_nearest_result)
                                k_distances = k_nearest_result{1};
                                k_nearest_q_vals = k_nearest_result{3};
                                k_indices = k_nearest_result{4};
                                distance_inverse = 1./k_distances;
                                q_vals = [];
                                for i = 1:D
                                    q_val = (distance_inverse*k_nearest_q_vals(:,i))/sum(distance_inverse);
                                    q_vals = [q_vals,q_val];
                                end
                                future_reward=future_reward+max(q_vals);
                            else
    %                             todo: add average information from the
    %                             learned table
                            end
                        end
                    end
                    
                    future_reward = future_reward/N_beliefs;
                    current_q_vals = [current_q_vals,in_reward+future_reward];
                end
end