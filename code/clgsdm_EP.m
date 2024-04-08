function [decision_results,information,forward_messages,backward_messages,states,state_mean_diff]=clgsdm_EP(initial_state,initial_mean,initial_var,transition_models,measurement_model,T,K,convergence_threshold,is_ADF,states_measurements,prest_decisions)

forward_messages = {};
backward_messages = {};
decision_results=zeros([1,T]);
initial_model = {initial_mean,initial_var};

states=[];
measurements=[];
R = measurement_model{3};
H = measurement_model{1};
b = measurement_model{2};
information = [];
max_marginal_vars = [];
for t=1:T
    current_max_d = 0;
    last_mean = initial_mean;
    last_var = initial_var;
    last_forward_message = {last_var\last_mean,inv(last_var)};
    if t~=1
       last_forward_message = forward_messages{t-1};
        last_var = inv(last_forward_message{2});
       last_mean = last_forward_message{2}\last_forward_message{1};
    end
        if isempty(states_measurements)
            current_max = -inf;
            current_max_marginal_var = 0;
            
        %     predict and decide
            for d=1:K
                    model = transition_models{d};
                    joint_model=directly_compute_moment_matching(model,measurement_model,last_mean,last_var);

                    [joint_mean,joint_cov]=compute_moments_of_gmm(joint_model);
                    marginal_var = joint_cov(1,1);
                    joint_precision = inv(joint_cov);
                    conditional_var = inv(joint_precision(1,1));
                    current_info_gain = 0.5*log(marginal_var)-0.5*log(conditional_var);
                    if current_info_gain>current_max
                        current_max=current_info_gain;
                        current_max_d = d;
                        current_max_marginal_var =marginal_var;
                    end
            end
                max_marginal_vars = [max_marginal_vars;current_max_marginal_var];
        else
            d = prest_decisions(t);
            model = transition_models{d};
            joint_model=directly_compute_moment_matching(model,measurement_model,last_mean,last_var);
            [joint_mean,joint_cov]=compute_moments_of_gmm(joint_model);
            marginal_var = joint_cov(1,1);
            max_marginal_vars = [max_marginal_vars;marginal_var];
            current_max_d = d;
            state_measurement = states_measurements{t};
            states = [states;state_measurement{1}];
            measurements = [measurements;state_measurement{2}];
            
        end
        decision_results(t)=current_max_d;
    %    simulate the dynamics and measurements
    if isempty(states_measurements)
    
        last_state = initial_state;
        if t~=1
            last_state = states(t-1);
        end
        next_state = dynamic_transition(last_state,current_max_d,transition_models);
        states=[states;next_state];
        measurement = measure(next_state,measurement_model);
        measurements=[measurements;measurement];
    end    
        
        
    %  the end of simulation
        
        model = transition_models{current_max_d};
        last_mean_cov = {initial_mean,inv(initial_var)};
        if t~=1
            last_message = forward_messages{t-1};
            last_mean_cov = {last_message{2}\last_message{1},last_message{2}};
        end
%         update alpha_t

        [forward_mean_cov_new,model_new]=update_after_measurement(model,measurement_model,measurements(t),last_mean_cov);
        transition_models{current_max_d}=model_new;
        forward_messages{t}={forward_mean_cov_new{2}\forward_mean_cov_new{1},inv(forward_mean_cov_new{2})};
        
        if is_ADF~=true
            if t>1
                converged = false;
                ite_num = 0;
                
                while ~converged
                    converged = true;
                    ite_num = ite_num+1;
    %                 update backward messages
                    for i=t-1:-1:1
                        measurement = measurements(i+1);
                        decision = decision_results(i);
                        model = transition_models{decision};
                        forward_message = forward_messages{i};

                        weights = model{1};
                        coefs = model{2};
                        mean_vec = model{3};
                        var_vec = model{4};

                        num_of_comp=length(weights);

                        computed_mean = mean_vec;
                        computed_var = var_vec;
                        eta_alpha_t = forward_message{1};
                        Lambda_alpha_t = forward_message{2};
                        mu_alpha_t = Lambda_alpha_t\eta_alpha_t;
                        if i~=t-1
                            backward_message=backward_messages{i+1};
                            eta_beta_t1 = backward_message{1};
                            Lambda_beta_t1 = backward_message{2};
                            mu_beta_t1 = Lambda_beta_t1\eta_beta_t1;

                            for j=1:num_of_comp
                                v_i = var_vec(j);
                                A_i = coefs(j);
                                a_i = mean_vec(j);
                                w_i = weights(j);

                                Lambda_i = inv(v_i)+H'/R*H;
                                L_i=Lambda_beta_t1+Lambda_i;
                                F_i = Lambda_i\v_i*Lambda_i;
                                f_i = Lambda_i\(H'/R*(measurement-b)+v_i\a_i);

                                d_i = F_i\(mu_beta_t1-f_i);
                                D_i = inv(F_i*Lambda_beta_t1*F_i')+inv(F_i*Lambda_i*F_i');
                                E_i = inv(D_i);
                                e_i = D_i\d_i;
                                M_i_inv = R+H*v_i*H';
                                HA_i = H*A_i;
                                S_i_inv = Lambda_alpha_t+HA_i'/M_i_inv*HA_i;
                                mu_i = S_i_inv\(HA_i'/M_i_inv*(measurement-H*a_i-b)+eta_alpha_t);
                                w_i_bar = log(w_i)+log(normpdf(measurement,HA_i*mu_alpha_t+H*a_i+b,sqrt(R+H*v_i*H'+HA_i/Lambda_alpha_t*HA_i')));
                                c_i_norm = normpdf(d_i,mu_i,sqrt(D_i+inv(S_i_inv)));
                                weight_new = w_i_bar + log(c_i_norm);
                                component_var_new = inv(S_i_inv+E_i);
                                component_mean_new = (S_i_inv+E_i)\(e_i+S_i_inv*mu_i);
                                weights(j)=weight_new;
                                computed_mean(j)=component_mean_new;
                                computed_var(j) = component_var_new;
                            end
                        else
                            for j=1:num_of_comp
                                v_i = var_vec(j);
                                A_i = coefs(j);
                                a_i = mean_vec(j);
                                w_i = weights(j);
                                HA_i = H*A_i;
                                weight_new = log(w_i)+log(normpdf(measurement,HA_i*mu_alpha_t+H*a_i+b,sqrt(inv(Lambda_alpha_t)+HA_i*(R+H*v_i*H')*HA_i')));
                                component_var_new = inv(Lambda_alpha_t+HA_i'/(R+H*v_i*H')*HA_i);
                                component_mean_new = component_var_new*(HA_i'/(R+H*v_i*H')*(measurement-H*a_i-b)+eta_alpha_t);


                                weights(j)=weight_new;
                                computed_mean(j)=component_mean_new;
                                computed_var(j) = component_var_new;
                            end
                        end
%                             pull the zero check out
                             if sum(exp(weights),'all')==0
                                 break;
                             elseif isnan(sum(exp(weights),'all'))
                                 break;
                             else
                                 weights = weights-max(weights);
                                 weights = exp(weights)./sum(exp(weights),'all');
                                 model{1}=weights;
                                 transition_models{decision} = model;
            %                     moment matching
                                 [posterior_mean,posterior_var] = compute_moments_of_gmm({weights,computed_mean,computed_var});
                                updated_pre = inv(posterior_var)-forward_message{2};
                                updated_mean = posterior_var\posterior_mean-forward_message{1};
                                updated_message = {updated_mean,updated_pre};
                                if i<=length(backward_messages)
                                    original_backward = backward_messages{i};
                                    mean_update_rate = det(abs(updated_mean-original_backward{1}))/det(abs(original_backward{1}));
                                    cov_update_rate = det(abs(updated_pre-original_backward{2}))/det(abs(original_backward{2}));
                                    if cov_update_rate>convergence_threshold||mean_update_rate>convergence_threshold
                                        converged=false;
                                    end
                                end

                                backward_messages{i}=updated_message;
                             end

                    end
%                     plot_result(t,states,measurements,forward_messages,backward_messages,'after backward before forward approximation results at time '+string(t)+', iteration '+string(ite_num));
    %                 update forward messages
                    for i=2:t
                        measurement = measurements(i);
                        decision = decision_results(i);
                        model = transition_models{decision};

                        last_forward_message = forward_messages{i-1};

                        posterior_model=compute_posterior(model,measurement_model,measurement,{last_forward_message{2}\last_forward_message{1},inv(last_forward_message{2})});
                        updated_message={};

                        if i==t
                            [forward_message_new,model_new] = update_after_measurement(model,measurement_model,measurement,{last_forward_message{2}\last_forward_message{1},inv(last_forward_message{2})});
                           updated_pre = inv(forward_message_new{2});
                           updated_mean = forward_message_new{2}\forward_message_new{1};
                           updated_message = {updated_mean,updated_pre};
                           forward_messages{i}=updated_message;
                           model{1} = model_new{1};
                        else

                            pos_mean_vec = posterior_model{2};
                            pos_var_vec = posterior_model{3};

                            weights=model{1};
                            coefs = model{2};
                            mean_vec = model{3};
                            var_vec = model{4};

                            num_of_comp=length(weights);
                            backward_message = backward_messages{i};
                            backward_eta = backward_message{1};
                            backward_Lambda = backward_message{2};
                            backward_var = inv(backward_Lambda);
                            backward_mean = backward_Lambda\backward_eta;
                            eta_alpha_t_1 = last_forward_message{1};
                            Lambda_alpha_t_1 = last_forward_message{2};
                            mu_alpha_t_1 = Lambda_alpha_t_1\eta_alpha_t_1;

                            for j=1:num_of_comp
                                pos_var = pos_var_vec(j);
                                pos_mean = pos_mean_vec(j);
                                pos_Lambda = inv(pos_var);
                                pos_eta = pos_var\pos_mean;
                                A_i = coefs(j);
                                a_i = mean_vec(j);
                                w_i = weights(j);
                                v_i = var_vec(j);

                                Lambda_zero = backward_Lambda+H'\R*H;
                                eta_zero = backward_eta+H'\R*(measurement-b);
                                mu_zero = Lambda_zero\eta_zero;

                                Lambda_i = inv(v_i+A_i*Lambda_alpha_t_1*A_i');
                                eta_i = Lambda_i*(A_i*mu_alpha_t_1+a_i);
                                mu_i = Lambda_i\eta_i;
                                weight_new = log(w_i)+log(normpdf(mu_i,mu_zero,sqrt(inv(Lambda_zero)+inv(Lambda_i))));
                                pos_mean_vec(j) = (Lambda_i+Lambda_zero)\(eta_i+eta_zero);
                                pos_var_vec(j) = inv(Lambda_i+Lambda_zero);
                                weights(j) = weight_new;
                            end
                            if sum(exp(weights),'all')==0
                                break;
                            elseif isnan(sum(exp(weights),'all'))
                                break;
                            else
                                weights = weights-max(weights);
                                 weights = exp(weights)./sum(exp(weights),'all');
                                model{1}=weights;
                                transition_models{decision} = model;
            %                     moment matching
                                [posterior_mean,posterior_var] = compute_moments_of_gmm({weights,pos_mean_vec,pos_var_vec});
                                updated_pre = inv(posterior_var)-backward_Lambda;
                                updated_mean = posterior_var\posterior_mean-backward_eta;
                                updated_message = {updated_mean,updated_pre};
                                forward_messages{i}=updated_message;
                            end

                        end

                        if i<=length(forward_messages)
                            original_forward = forward_messages{i};
                            updated_mean = updated_message{1};
                            updated_pre = updated_message{2};
                            mean_update_rate = det(abs(updated_mean-original_forward{1}))/det(abs(original_forward{1}));
                            cov_update_rate = det(abs(updated_pre-original_forward{2}))/det(abs(original_forward{2}));
                            if cov_update_rate>convergence_threshold||mean_update_rate>convergence_threshold
                                converged=false;
                            end
                        end
                    end
%                     plot_result(t,states,measurements,forward_messages,backward_messages,'approximation results at time '+string(t)+', iteration '+string(ite_num));
                    
                end
            end    
        end
        
end

state_mean_diff = [];
for t=1:T
    forward_message = forward_messages{t};
    max_marginal_cov = max_marginal_vars(t);
    posterior_cov = inv(forward_message{2});
   posterior_mean = forward_message{2}\forward_message{1};
    if t~=T
        
        if is_ADF~=true
            backward_message = backward_messages{t};
            posterior_cov = inv(forward_message{2}+backward_message{2});
            posterior_mean = (forward_message{2}+backward_message{2})\(forward_message{1}+backward_message{1});
        end
        
        information_gain = 0.5*log(det(max_marginal_cov))-0.5*log(det(posterior_cov));
        information = [information;information_gain];
    else
        
        information_gain = 0.5*log(det(max_marginal_cov))-0.5*log(det(posterior_cov));
        information = [information;information_gain];
    end
    state_mean_diff= [state_mean_diff;abs(posterior_mean-states(t))];
end



end




