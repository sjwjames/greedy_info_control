function states_measurements = generate_GMM_states_measurements(initial_state,dynamic_trainsition_models,measurement_model,T,decisions)
% hard code for scalar
states_measurements = zeros([T,2]);
for t=1:T
    decision = decisions(t);
    last_state = initial_state;
    if t~=1
       last_state = states_measurements(t-1,1);
    end
    next_state = dynamic_transition(last_state,decision,dynamic_trainsition_models); 
    next_measurement = measure(next_state,measurement_model);
    states_measurements(t,1) = next_state;
    states_measurements(t,2) = next_measurement;
end

end