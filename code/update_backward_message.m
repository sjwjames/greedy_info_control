function [backward_messages,converged,steps_skipped] = update_backward_message(decisions,measurements,forward_messages,backward_messages,transition_models,measurement_model,t,convergence_threshold,step_size)
    converged =false;
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
    steps_skipped = 0;
    for i=t-1:-1:1
        measurement = measurements(i+1);
        decision = decisions(i);
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
%         if Lambda_alpha_t<0
%             steps_skipped= steps_skipped+1;
%             continue;
%         end
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
               L_i=inv(Lambda_beta_t1)+inv(Lambda_i);
               F_i = inv(Lambda_i)*inv(v_i)*A_i;
               inv_Fi = inv(F_i);
               f_i = Lambda_i\(H'/R*(measurement-b)+v_i\a_i);

               d_i = F_i\(mu_beta_t1-f_i);
               D_i = F_i\L_i*inv_Fi';
               E_i = inv(D_i);
               e_i = D_i\d_i;
               M_i_inv = R+H*v_i*H';
               HA_i = H*A_i;
               S_i_inv = Lambda_alpha_t+HA_i'/M_i_inv*HA_i;
               mu_i = S_i_inv\(HA_i'/M_i_inv*(measurement-H*a_i-b)+eta_alpha_t);
               w_i_bar = w_i*normpdf(measurement,HA_i*mu_alpha_t+H*a_i+b,sqrt(R+H*v_i*H'+HA_i/Lambda_alpha_t*HA_i'));
               c_i_norm = normpdf(d_i,mu_i,sqrt(D_i+inv(S_i_inv)));
               weight_new = w_i_bar * c_i_norm;
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
                weight_new = w_i*normpdf(measurement,HA_i*mu_alpha_t+H*a_i+b,sqrt(inv(Lambda_alpha_t)+HA_i*(R+H*v_i*H')*HA_i'));
                component_var_new = inv(Lambda_alpha_t+HA_i'/(R+H*v_i*H')*HA_i);
                component_mean_new = component_var_new*(HA_i'/(R+H*v_i*H')*(measurement-H*a_i-b)+eta_alpha_t);


                weights(j)=weight_new;
                computed_mean(j)=component_mean_new;
                computed_var(j) = component_var_new;
             end
        end
             if sum(exp(weights),'all')==0
                 steps_skipped= steps_skipped+1;
                continue;
             elseif isnan(sum(exp(weights),'all'))
                 steps_skipped= steps_skipped+1;
                continue;
             else
                weights = weights./sum(weights);

            %                     moment matching
                [posterior_mean,posterior_var] = compute_moments_of_gmm({weights,computed_mean,computed_var});
                current_backward_message = backward_messages{i};
                updated_pre = step_size*(inv(posterior_var)-forward_message{2})+(1-step_size)*current_backward_message{2};
%                 if updated_pre<=0
% %                     updated_pre = 0.0001;
%                     continue;
%                 end
                updated_mean = step_size*(posterior_var\posterior_mean-forward_message{1})+(1-step_size)*current_backward_message{1};
%                 updated_pre = inv(posterior_var);
%                 updated_mean = posterior_var\posterior_mean;
                updated_message = {updated_mean,updated_pre};
                if i<=length(backward_messages)
                   original_backward = backward_messages{i};
                   if ~isempty(original_backward)
                       mean_update_rate = abs(updated_mean-original_backward{1});
                       cov_update_rate = abs(updated_pre-original_backward{2});
                       if cov_update_rate<=convergence_threshold&&mean_update_rate<=convergence_threshold
                        converged=true;
                       end
                   else
                       converged = false;
                   end
                   
                       
                 end

               backward_messages{i}=updated_message;
             end
    end
end