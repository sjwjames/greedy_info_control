function plot_discretization_predictions(state_dist,discretized_states,discretized_measurements,pred_dists,posterior,d,current_t)
    pred_state_dist = pred_dists{1};
    pred_likelihood  = pred_dists{2};
    pred_joint = pred_dists{3};
    measurement_likelihood = pred_dists{4};
%     t = tiledlayout(1,1);
%     ax1 = axes(t);
    
%     ax1.XLabel = "X";
%     ax1.YLabel = "Probablity Density";
    
    x1 = discretized_states;
    plot(x1,state_dist,'-b',x1,pred_state_dist,'-g',x1,posterior,'-r',x1,measurement_likelihood,'-k')
    hold on
    xlabel('States')
    ylabel('Probability density')
%     ax2 = axes(t);
%     ax2.XAxisLocation = 'left';
%     ax2.YAxisLocation = 'top';
%     plot(ax2,discretized_measurements,pred_likelihood,'-r')
%     hold on
%     ax2.XLabel = "Y";
%     ax2.YLabel = "Probablity Density";

%     ax3 = axes(t);
    yyaxis right
    contour(discretized_states,discretized_measurements,pred_joint)
    ylabel('Measurements')
    
    legend("prior","predictive marginal","Posterior","Measurement distribution","predictive joint",'Location','northeast','FontSize', 12)

%     legend("prior","predictive marginal","predictive marginal likelihood","predictive joint",'Location','northwest','FontSize', 12)
    
    % ax2.Color = 'none';
%     ax1.Box = 'off';
%     ax2.Box = 'off';
%     ax3.Box = 'off';
    title("Discretization distributions,t="+string(current_t)+"d="+string(d),'FontSize', 14)
    hold off
    saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/discretization distributions,t='+string(current_t)+'d='+string(d)+'.png');
    clf
end