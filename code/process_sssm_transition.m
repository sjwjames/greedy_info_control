function [s,x]=process_sssm_transition(current_s,current_x,decision,sssm)
    rho = sssm{1};
    discrete_transition = rho(current_s,:);
    s = find(mnrnd(1,discrete_transition)==1);
    
    transition_model = sssm{2}{s}{decision};
    x = normrnd(transition_model{1}*current_x+transition_model{2},sqrt(transition_model{3}));
end