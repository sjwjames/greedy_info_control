function plot_result(time_step,experiment_result_collection,preset_title,color_assigned,result_file_directory,legend_location)
means = experiment_result_collection{1};
stds = experiment_result_collection{2};
% EP_means = experiment_result_collection{3};
% EP_stds = experiment_result_collection{4};
discretized_bins = experiment_result_collection{3};
state_dist_at_steps = experiment_result_collection{4};
true_states = experiment_result_collection{5};
measurements = experiment_result_collection{6};
method_name = experiment_result_collection{7};
line_width = 3;


if method_name=="C-POMDP"
%     X = -25:25;
    X = 1:time_step;
    figure,
    colormap(gray);
    imagesc(1:time_step, discretized_bins,1-state_dist_at_steps');
    hold on
    plot(X,true_states,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
    hold on
    scatter(X,measurements,14,'*','LineWidth',line_width)
    hold on
    plot(X,means,'--xr','MarkerSize',10,'LineWidth',line_width,'color',color_assigned)
    % hold on
    % plot(X,EP_means,'--ob','LineWidth',line_width)
    hold on
    % alpha(f1,0.7)
    % alpha(f2,0.7)
    
    set(gca,'YDir','normal') 
    
    % method_variance = method_name+" Variance";
    % method_means =  method_name+" means";
    legend('True States','Measurements',method_name+" estimation",'Location',legend_location,'FontSize', 16)
    % legend("ADF Variance","EP Variance",'True States','Measurements',"ADF means","EP means",'Location','northwest')
    
%     title(preset_title)
    ylabel('States and Measurements','FontSize', 18)
    xlabel('Timesteps','FontSize', 18)
    hold off
    exportgraphics(gcf,result_file_directory+preset_title+'.pdf');
else
    X=1:time_step;
    % X2 = [X,fliplr(X)];
    % inbetween = [means+stds,fliplr(means-stds)];
    % EP_inbetween = [EP_means+EP_stds,fliplr(EP_means-EP_stds)];
    figure,
    colormap(gray);
    imagesc(1:time_step, discretized_bins,1-state_dist_at_steps');
    hold on
    % f1 = fill(X2,inbetween, color_assigned);
    % hold on
    % f2 = fill(X2,EP_inbetween, [0.94 0.94 1]);
    % hold on
    plot(X,true_states,'-o','MarkerSize',10,'color','g','LineWidth',line_width)
    hold on
    scatter(X,measurements,14,'*','LineWidth',line_width)
    hold on
    plot(X,means,'--xr','MarkerSize',10,'LineWidth',line_width,'color',color_assigned)
    % hold on
    % plot(X,EP_means,'--ob','LineWidth',line_width)
    hold on
    % alpha(f1,0.7)
    % alpha(f2,0.7)
    
    set(gca,'YDir','normal') 
    
    % method_variance = method_name+" Variance";
    % method_means =  method_name+" means";
    legend('True States','Measurements',method_name+" estimation",'Location',legend_location,'FontSize', 16)
    % legend("ADF Variance","EP Variance",'True States','Measurements',"ADF means","EP means",'Location','northwest')
    
    title(preset_title)
    xlabel('Timesteps','FontSize', 18)
    ylabel('States','FontSize', 18)
    hold off
    exportgraphics(gcf,result_file_directory+preset_title+'.pdf');
end

end