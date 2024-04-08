function [state_range,measurement_range] = generate_joint_ranges(K,initial_state,transition_models,measurement_model,time_threshold)
state_range = [initial_state,initial_state];
measurement_coef = measurement_model{1};
measurement_noise_mean = measurement_model{2};
measurement_noise_var = measurement_model{3};
initial_measurement = measure(initial_state,measurement_model);
measurement_range = [initial_measurement,initial_measurement];


models = transition_models;
t=0;
states=[initial_state];
measurements = [initial_measurement];
current_state = initial_state;

while t<time_threshold
    decision = randi([1,K]);

    next_state = dynamic_transition(current_state,decision,models);
    if next_state<state_range(1)
        state_range(:,1)=next_state;
    elseif next_state>state_range(2)
        state_range(:,2) = next_state;
    end
    current_state = next_state;
    states=[states,current_state];
    current_measurement = measure(current_state,measurement_model);    
    measurements=[measurements,current_measurement];
    if current_measurement<measurement_range(1)
        measurement_range(:,1)=current_measurement;
    elseif current_measurement>measurement_range(2)
        measurement_range(:,2) = current_measurement;
    end


    t=t+1;
end
% figure,
% plot([0:t],states)
% hold on
% plot([0:t],measurements)
% legend('States','Measurements')
% xlabel('Timesteps')
% hold off
end