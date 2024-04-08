function [smoothed_means,smoothed_variances]=compute_smoothed_moments(forward_messages,backward_messages,T)
smoothed_means = [];
smoothed_variances = [];


for t=1:T
    forward_message = forward_messages{t};
    forward_mean = forward_message{2}\forward_message{1};
    forward_variance = inv(forward_message{2});
    if t~=T
        backward_message = backward_messages{t};
        smoothed_mean = (forward_message{2}+backward_message{2})\(forward_message{1}+backward_message{1});
        smoothed_variance = inv((forward_message{2}+backward_message{2}));
        smoothed_means = [smoothed_means,smoothed_mean];
        smoothed_variances = [smoothed_variances,smoothed_variance];
    else
        smoothed_means = [smoothed_means,forward_mean];
        smoothed_variances = [smoothed_variances,forward_variance];
    end
end
end