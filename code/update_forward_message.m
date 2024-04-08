function [forward_messages,converged,steps_skipped]=update_forward_message(decisions,measurements,forward_messages,backward_messages,transition_models,initial_model,measurement_model,t,convergence_threshold,step_size)
    converged = false;
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
    steps_skipped = 0;
    for i=1:t
        measurement = measurements(i);
        decision = decisions(i);
        model = transition_models{decision};
        last_forward_message = {initial_model{2}\initial_model{1},inv(initial_model{2})};
        if i~=1
            last_forward_message = forward_messages{i-1};
        end
        

        posterior_model=compute_posterior(model,measurement_model,measurement,{last_forward_message{2}\last_forward_message{1},inv(last_forward_message{2})});
        updated_message={};

        if i>length(backward_messages)
            [forward_message_new] = update_after_measurement(model,measurement_model,measurement,{last_forward_message{2}\last_forward_message{1},inv(last_forward_message{2})});
            updated_pre = inv(forward_message_new{2});
            updated_mean = forward_message_new{2}\forward_message_new{1};
            updated_message = {updated_mean,updated_pre};
            forward_messages{i}=updated_message;

        else

            pos_mean_vec = posterior_model{2};
            pos_var_vec = posterior_model{3};

            weights=posterior_model{1};
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
%             if backward_Lambda<0
%                 steps_skipped= steps_skipped+1;
%                 continue;
%             end
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

                Lambda_i = inv(v_i+A_i/Lambda_alpha_t_1*A_i');
                eta_i = Lambda_i*(A_i*mu_alpha_t_1+a_i);
                mu_i = Lambda_i\eta_i;

                weight_new = w_i*normpdf(mu_i,mu_zero,sqrt(inv(Lambda_zero)+inv(Lambda_i)));
                pos_mean_vec(j) = (Lambda_i+Lambda_zero)\(eta_i+eta_zero);
                pos_var_vec(j) = inv(Lambda_i+Lambda_zero);
                weights(j) = weight_new;
            end
            if sum(exp(weights),'all')==0
                steps_skipped= steps_skipped+1;
               continue;
            elseif isnan(sum(exp(weights),'all'))
                steps_skipped= steps_skipped+1;
               continue;
            else
               weights = weights./sum(weights);
%                model{1}=weights;
%                transition_models{decision} = model;
            %  moment matching
               [posterior_mean,posterior_var] = compute_moments_of_gmm({weights,pos_mean_vec,pos_var_vec});
               current_forward_message = forward_messages{i};
               updated_pre = step_size*(inv(posterior_var)-backward_Lambda)+(1-step_size)*current_forward_message{2};
%                if updated_pre<=0
% %                     updated_pre = 0.0001;
%                     continue;
%                 end
               updated_mean = step_size*(posterior_var\posterior_mean-backward_eta)+(1-step_size)*current_forward_message{1};
               updated_message = {updated_mean,updated_pre};
               forward_messages{i}=updated_message;
            end

        end

            if i<=length(forward_messages)
               original_forward = forward_messages{i};
               updated_mean = updated_message{1};
               updated_pre = updated_message{2};
               mean_update_rate = abs(updated_mean-original_forward{1});
               cov_update_rate = abs(updated_pre-original_forward{2});
                if cov_update_rate<=convergence_threshold&&mean_update_rate<=convergence_threshold
                    converged=true;
                end
            end
     end
end