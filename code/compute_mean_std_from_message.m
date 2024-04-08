function [means,stds]=compute_mean_std_from_message(forward_messages,backward_messages,T)
    means=[];
    stds = [];
    
    
    for t=1:T
        forward_message = forward_messages{t};
        forward_mean = forward_message{2}\forward_message{1};
        forward_std = inv(forward_message{2});
        if isempty(backward_messages)
            means = [means,forward_mean];
            stds = [stds,forward_std];
        else
            if t~=T
                backward_message = backward_messages{t};
                forward_backward_mean = (forward_message{2}+backward_message{2})\(forward_message{1}+backward_message{1});
                forward_backward_std = inv((forward_message{2}+backward_message{2}))^0.5;
                means = [means,forward_backward_mean];
                stds = [stds,forward_backward_std];
            else
                means = [means,forward_mean];
                stds = [stds,forward_std];
            end
        end
        
    end

end