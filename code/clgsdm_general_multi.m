function [forward_messages,backward_messages,ADF_messages,forward_pass_skipped_intotal,backward_pass_skipped_intotal,ite_num]=clgsdm_general_multi(initial_model,transition_models,measurement_model,T,convergence_threshold,measurements,prest_decisions,preset_forward_messages,preset_backward_messages,state_dim)
    forward_messages = preset_forward_messages;
    backward_messages = preset_backward_messages;
    if isempty(forward_messages)&&isempty(backward_messages)
        for t=1:T
            forward_messages{t}={zeros([1,state_dim]),zeros([state_dim,state_dim])};
            backward_messages{t}={zeros([1,state_dim]),zeros([state_dim,state_dim])};
        end
    end
    

    converged = false;
    ite_num = 0;
    ADF_messages = {};

    step_size = 0.1;
    forward_pass_skipped_intotal = 0;
    backward_pass_skipped_intotal = 0;
    while ~converged
        ite_num = ite_num+1;
        [forward_messages,forward_converged,forward_pass_skipped]=update_forward_message_multi(prest_decisions,measurements,forward_messages,backward_messages,transition_models,initial_model,measurement_model,T,convergence_threshold,step_size,state_dim);
        if ite_num==1
            ADF_messages = forward_messages;
        end
        if T==1
            backward_converged = 1;
            backward_pass_skipped = 0;
        else
            [backward_messages,backward_converged,backward_pass_skipped]=update_backward_message_multi(prest_decisions,measurements,forward_messages,backward_messages,transition_models,measurement_model,T,convergence_threshold,step_size,state_dim);
        end
        
            
                if forward_converged&&backward_converged
                    converged = true;
                end

             forward_pass_skipped_intotal = forward_pass_skipped_intotal+forward_pass_skipped;
             backward_pass_skipped_intotal = backward_pass_skipped_intotal+backward_pass_skipped;
            if ite_num>100000
                return
            end
    end
    
end

function [forward_messages,converged,steps_skipped]=update_forward_message_multi(decisions,measurements,forward_messages,backward_messages,transition_models,initial_model,measurement_model,t,convergence_threshold,step_size,state_dim)
    converged = false;
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
    steps_skipped = 0;
    for i=1:t
        measurement = measurements(i,:);
        decision = decisions(i);
        model = transition_models{decision};
        last_forward_message = {(initial_model{2}\initial_model{1}')',inv(initial_model{2})};
        if i~=1
            last_forward_message = forward_messages{i-1};
        end
        

        posterior_model=compute_posterior_multi(model,measurement_model,measurement,{(last_forward_message{2}\last_forward_message{1}')',inv(last_forward_message{2})},state_dim);
        updated_message={};

        if i>length(backward_messages)
            [forward_message_new] = update_after_measurement_multi(model,measurement_model,measurement,{(last_forward_message{2}\last_forward_message{1}')',inv(last_forward_message{2})});
            updated_pre = inv(forward_message_new{2});
            updated_mean = (forward_message_new{2}\forward_message_new{1}')';
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
            backward_mean = (backward_Lambda\backward_eta')';
            eta_alpha_t_1 = last_forward_message{1};
            Lambda_alpha_t_1 = last_forward_message{2};
            mu_alpha_t_1 = (Lambda_alpha_t_1\eta_alpha_t_1')';
%             if backward_Lambda<0
%                 steps_skipped= steps_skipped+1;
%                 continue;
%             end
            for j=1:num_of_comp
                pos_var = pos_var_vec(:,:,j);
                pos_mean = pos_mean_vec(j,:);
                pos_Lambda = inv(pos_var);
                pos_eta = (pos_var\pos_mean')';
                A_i = coefs(:,:,j);
                a_i = mean_vec(j,:);
                w_i = weights(j);
                v_i = var_vec(:,:,j);

                Lambda_zero = backward_Lambda+H'\R*H;
                eta_zero = backward_eta+(measurement-b)*(H'\R)';
                mu_zero = (Lambda_zero\eta_zero')';

                Lambda_i = inv(v_i+A_i/Lambda_alpha_t_1*A_i');
                eta_i = (mu_alpha_t_1*A_i'+a_i)*Lambda_i';
                mu_i = (Lambda_i\eta_i')';

                weight_new = w_i*mvnpdf(mu_i,mu_zero,inv(Lambda_zero)+inv(Lambda_i));
                pos_mean_vec(j,:) = ((Lambda_i+Lambda_zero)\(eta_i+eta_zero)')';
                pos_var_vec(:,:,j) = inv(Lambda_i+Lambda_zero);
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
               [posterior_mean,posterior_var] = compute_moments_of_gmm_multi({weights,pos_mean_vec,pos_var_vec},state_dim);
               current_forward_message = forward_messages{i};
               updated_pre = step_size*(inv(posterior_var)-backward_Lambda)+(1-step_size)*current_forward_message{2};
%                if updated_pre<=0
% %                     updated_pre = 0.0001;
%                     continue;
%                 end
               updated_mean = step_size*((posterior_var\posterior_mean')'-backward_eta)+(1-step_size)*current_forward_message{1};
               updated_message = {updated_mean,updated_pre};
               forward_messages{i}=updated_message;
            end

        end

            if i<=length(forward_messages)
               original_forward = forward_messages{i};
               updated_mean = updated_message{1};
               updated_pre = updated_message{2};
               mean_update_rate = norm(updated_mean-original_forward{1});
               cov_update_rate = norm(updated_pre-original_forward{2});
                if cov_update_rate<=convergence_threshold&&mean_update_rate<=convergence_threshold
                    converged=true;
                end
            end
     end
end

function [backward_messages,converged,steps_skipped] = update_backward_message_multi(decisions,measurements,forward_messages,backward_messages,transition_models,measurement_model,t,convergence_threshold,step_size,state_dim)
    converged =false;
    R = measurement_model{3};
    H = measurement_model{1};
    b = measurement_model{2};
    steps_skipped = 0;
    for i=t-1:-1:1
        measurement = measurements(i+1,:);
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
        mu_alpha_t = (Lambda_alpha_t\eta_alpha_t')';
%         if Lambda_alpha_t<0
%             steps_skipped= steps_skipped+1;
%             continue;
%         end
        if i~=t-1
           backward_message=backward_messages{i+1};
           eta_beta_t1 = backward_message{1};
           Lambda_beta_t1 = backward_message{2};
           mu_beta_t1 = (Lambda_beta_t1\eta_beta_t1')';

           for j=1:num_of_comp
               v_i = var_vec(:,:,j);
               A_i = coefs(:,:,j);
               a_i = mean_vec(j,:);
               w_i = weights(j);

               Lambda_i = inv(v_i)+H'/R*H;
               L_i=inv(Lambda_beta_t1)+inv(Lambda_i);
               F_i = inv(Lambda_i)*inv(v_i)*A_i;
               inv_Fi = inv(F_i);
               f_i = (Lambda_i\((measurement-b)*(H'/R)'+(v_i\a_i')')')';

               d_i = (F_i\(mu_beta_t1-f_i)')';
               D_i = F_i\L_i*inv_Fi';
               E_i = inv(D_i);
               e_i = (D_i\d_i')';
               M_i_inv = R+H*v_i*H';
               HA_i = H*A_i;
               S_i_inv = Lambda_alpha_t+HA_i'/M_i_inv*HA_i;
               mu_i = (S_i_inv\((measurement-a_i*H'-b)*(HA_i'/M_i_inv)'+eta_alpha_t)')';
               w_i_bar = w_i*mvnpdf(measurement,mu_alpha_t*HA_i'+a_i*H'+b,R+H*v_i*H'+HA_i/Lambda_alpha_t*HA_i');
               disp(D_i+inv(S_i_inv));
               c_i_norm = mvnpdf(d_i,mu_i,D_i+inv(S_i_inv));
               weight_new = w_i_bar * c_i_norm;
               component_var_new = inv(S_i_inv+E_i);
               component_mean_new = ((S_i_inv+E_i)\(e_i+mu_i*S_i_inv')')';
               weights(j)=weight_new;
               computed_mean(j,:)=component_mean_new;
               computed_var(:,:,j) = component_var_new;
            end
       else
            for j=1:num_of_comp
                v_i = var_vec(:,:,j);
                A_i = coefs(:,:,j);
                a_i = mean_vec(j,:);
                w_i = weights(j);
                HA_i = H*A_i;
                weight_new = w_i*mvnpdf(measurement,mu_alpha_t*HA_i'+a_i*H'+b,inv(Lambda_alpha_t)+HA_i*(R+H*v_i*H')*HA_i');
                component_var_new = inv(Lambda_alpha_t+HA_i'/(R+H*v_i*H')*HA_i);
                component_mean_new = ((measurement-a_i*H'-b)*(HA_i'/(R+H*v_i*H'))'+eta_alpha_t)*component_var_new';


                weights(j)=weight_new;
                computed_mean(j,:)=component_mean_new;
                computed_var(:,:,j) = component_var_new;
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
                [posterior_mean,posterior_var] = compute_moments_of_gmm_multi({weights,computed_mean,computed_var},state_dim);
                current_backward_message = backward_messages{i};
                updated_pre = step_size*(inv(posterior_var)-forward_message{2})+(1-step_size)*current_backward_message{2};
%                 if updated_pre<=0
% %                     updated_pre = 0.0001;
%                     continue;
%                 end
                updated_mean = step_size*((posterior_var\posterior_mean')'-forward_message{1})+(1-step_size)*current_backward_message{1};
%                 updated_pre = inv(posterior_var);
%                 updated_mean = posterior_var\posterior_mean;
                updated_message = {updated_mean,updated_pre};
                if i<=length(backward_messages)
                   original_backward = backward_messages{i};
                   if ~isempty(original_backward)
                       mean_update_rate = norm(updated_mean-original_backward{1});
                       cov_update_rate = norm(updated_pre-original_backward{2});
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