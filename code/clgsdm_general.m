function [forward_messages,backward_messages,ADF_messages,forward_pass_skipped_intotal,backward_pass_skipped_intotal,ite_num]=clgsdm_general(initial_model,transition_models,measurement_model,T,convergence_threshold,measurements,prest_decisions,preset_forward_messages,preset_backward_messages)
    forward_messages = preset_forward_messages;
    backward_messages = preset_backward_messages;
    if isempty(forward_messages)&&isempty(backward_messages)
        for t=1:T
            forward_messages{t}={0,0};
            backward_messages{t}={0,0};
        end
    end
    

    converged = false;
    ite_num = 0;
    ADF_messages = {};
    smoothed_means = [];
    smoothed_variances = [];
    step_size = 0.1;
    forward_pass_skipped_intotal = 0;
    backward_pass_skipped_intotal = 0;
    while ~converged
        ite_num = ite_num+1;
        [forward_messages,forward_converged,forward_pass_skipped]=update_forward_message(prest_decisions,measurements,forward_messages,backward_messages,transition_models,initial_model,measurement_model,T,convergence_threshold,step_size);
        if ite_num==1
            ADF_messages = forward_messages;
        end
        if T==1
            backward_converged = 1;
            backward_pass_skipped = 0;
        else
            [backward_messages,backward_converged,backward_pass_skipped]=update_backward_message(prest_decisions,measurements,forward_messages,backward_messages,transition_models,measurement_model,T,convergence_threshold,step_size);
        end
        
            
%              if ite_num<100
                if forward_converged&&backward_converged
                    converged = true;
                end
%             else
%                 
%              end
%                 [current_smoothed_means,current_smoothed_variances]=compute_smoothed_moments(forward_messages,backward_messages,T);
%                 if isempty(smoothed_means)
%                     smoothed_means_diffs = current_smoothed_means;
%                 else
%                     smoothed_means_diffs = abs(current_smoothed_means-smoothed_means);
%                 end
%                 if isempty(smoothed_variances)
%                     smoothed_variances_diffs = current_smoothed_variances;
%                 else
%                     smoothed_variances_diffs = abs(current_smoothed_variances-smoothed_variances);
%                 end
%                if max(smoothed_means_diffs)<convergence_threshold&&max(smoothed_variances_diffs)<convergence_threshold
%                     converged = true;
%                 end
%                 smoothed_variances = current_smoothed_variances;
%                 smoothed_means = current_smoothed_means;
             forward_pass_skipped_intotal = forward_pass_skipped_intotal+forward_pass_skipped;
             backward_pass_skipped_intotal = backward_pass_skipped_intotal+backward_pass_skipped;
            if ite_num>100000
                return
            end
%         plot_result(T,states,measurements,forward_messages,backward_messages,ADF_messages,'approximation results at time '+string(T)+', iteration '+string(ite_num),[],{});
    end
    
end