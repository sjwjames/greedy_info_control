function generate_gmmg_runtime_stats()
    n_range = 2:8;
    k_range = 5:10;
    ADF_runtime_means = zeros([length(n_range),length(k_range)]);
    PF_runtime_means = zeros([length(n_range),length(k_range)]);
    discretization_runtime_means = zeros([length(n_range),length(k_range)]);
    EP_runtime_means = zeros([length(n_range),length(k_range)]);
    ADF_GMM_runtime_means = zeros([length(n_range),length(k_range)]);
    ADF_runtime_stds = zeros([length(n_range),length(k_range)]);
    PF_runtime_stds = zeros([length(n_range),length(k_range)]);
    discretization_runtime_stds = zeros([length(n_range),length(k_range)]);
    EP_runtime_std = zeros([length(n_range),length(k_range)]);
    ADF_GMM_runtime_stds = zeros([length(n_range),length(k_range)]);
    save_location = "experiment_results/greedy ep/online decisions/different_decisions/runtime_stats/";
    for i=1:length(n_range)
        num_of_comp = n_range(i);
        for j=1:length(k_range)
            K = k_range(j);
            result_file_directory = "experiment_results/greedy ep/online decisions/different_decisions/n="+string(num_of_comp)+",k="+string(K)+"/";
            adf_time = readmatrix(result_file_directory+"ADF_simple_running_time.csv");
            ADF_runtime_means(i,j) = mean(adf_time,"all");
            ADF_runtime_stds(i,j) = std(adf_time,0,"all");
            discretize_time = readmatrix(result_file_directory+"discretization_running_time.csv");
            discretization_runtime_means(i,j) = mean(discretize_time,"all");
            discretization_runtime_stds(i,j) = std(discretize_time,0,"all");
            pf_time = readmatrix(result_file_directory+"pf_running_time.csv");
            PF_runtime_means(i,j) = mean(pf_time,"all");
            PF_runtime_stds(i,j) = std(pf_time,0,"all");
            EP_time = readmatrix(result_file_directory+"EP_running_time.csv");
            EP_runtime_means(i,j) = mean(EP_time,"all");
            EP_runtime_std(i,j) = std(EP_time,0,"all");
            ADF_GMM_time = readmatrix(result_file_directory+"ADF_gmm_unc_running_time.csv");
            ADF_GMM_runtime_means(i,j) = mean(ADF_GMM_time,"all");
            ADF_GMM_runtime_stds(i,j) = std(ADF_GMM_time,0,"all");
        end
        X = k_range;
        Y = [ADF_runtime_means(i,:);discretization_runtime_means(i,:);PF_runtime_means(i,:);EP_runtime_means(i,:);ADF_GMM_runtime_means(i,:)];
        errors = [ADF_runtime_stds(i,:);discretization_runtime_stds(i,:);PF_runtime_stds(i,:);EP_runtime_std(i,:);ADF_GMM_runtime_stds(i,:)];
        figure_title = "Decision time,"+string(num_of_comp)+" components in dynamic transition";
        figure,
%         errorbar(X,Y(1,:),errors(1,:),"-s","MarkerSize",10,'LineWidth',1,'Color','r')
%         hold on
%         errorbar(X,Y(2,:),errors(2,:),"-s","MarkerSize",10,'LineWidth',1,'Color','k')
%         hold on
%         errorbar(X,Y(3,:),errors(3,:),"-s","MarkerSize",10,'LineWidth',1,'Color','y')
%         hold on
%         errorbar(X,Y(4,:),errors(4,:),"-s","MarkerSize",10,'LineWidth',1,'Color','b')
%         hold on
%         errorbar(X,Y(5,:),errors(5,:),"-s","MarkerSize",10,'LineWidth',1,'Color','m')
%         hold off
        plot(X,Y(1,:),"-s","MarkerSize",10,'LineWidth',1,'Color','r')
        hold on
        plot(X,Y(2,:),"-s","MarkerSize",10,'LineWidth',1,'Color','k')
        hold on
        plot(X,Y(3,:),"-s","MarkerSize",10,'LineWidth',1,'Color','y')
        hold on
        plot(X,Y(4,:),"-s","MarkerSize",10,'LineWidth',1,'Color','b')
        hold on
        plot(X,Y(5,:),"-s","MarkerSize",10,'LineWidth',1,'Color','m')
        hold off
        set(gca,'xtick',X)
        set(gca,'YScale','log')
        legend("ADF-Gaussian","Numerical","PF","EP-Gaussian","ADF-GMM",'Location','northwest','FontSize', 12)
        xlabel('Number of decisions','FontSize', 14)
        ylabel('Decision time per step (log(s))','FontSize', 14)
        title(figure_title,'FontSize', 14)
        exportgraphics(gcf,save_location+figure_title+'.pdf')
    end
    

    for j=1:length(k_range)
        K = k_range(j);
        X = n_range;
        Y = [ADF_runtime_means(:,j)';discretization_runtime_means(:,j)';PF_runtime_means(:,j)';EP_runtime_means(:,j)';ADF_GMM_runtime_means(:,j)'];
        errors = [ADF_runtime_stds(:,j)';discretization_runtime_stds(:,j)';PF_runtime_stds(:,j)';EP_runtime_std(:,j)';ADF_GMM_runtime_stds(:,j)'];
        figure_title = "Decision time,"+string(K)+" decisions";
        figure,
%         errorbar(X,Y(1,:),errors(1,:),"-s","MarkerSize",10,'LineWidth',1,'Color','r')
%         hold on
%         errorbar(X,Y(2,:),errors(2,:),"-s","MarkerSize",10,'LineWidth',1,'Color','k')
%         hold on
%         errorbar(X,Y(3,:),errors(3,:),"-s","MarkerSize",10,'LineWidth',1,'Color','y')
%         hold on
%         errorbar(X,Y(4,:),errors(4,:),"-s","MarkerSize",10,'LineWidth',1,'Color','b')
%         hold on
%         errorbar(X,Y(5,:),errors(5,:),"-s","MarkerSize",10,'LineWidth',1,'Color','m')
%         hold off
        plot(X,Y(1,:),"-s","MarkerSize",10,'LineWidth',1,'Color','r')
        hold on
        plot(X,Y(2,:),"-s","MarkerSize",10,'LineWidth',1,'Color','k')
        hold on
        plot(X,Y(3,:),"-s","MarkerSize",10,'LineWidth',1,'Color','y')
        hold on
        plot(X,Y(4,:),"-s","MarkerSize",10,'LineWidth',1,'Color','b')
        hold on
        plot(X,Y(5,:),"-s","MarkerSize",10,'LineWidth',1,'Color','m')
        hold off
        set(gca,'xtick',X)
        set(gca,'YScale','log')
        legend("ADF-Gaussian","Numerical","PF","EP-Gaussian","ADF-GMM",'Location','northwest','FontSize', 12)
        xlabel('Number of components in dynamic transitions','FontSize', 14)
        ylabel('Decision time per step (log(s))','FontSize', 14)
        title(figure_title,'FontSize', 14)
        exportgraphics(gcf,save_location+figure_title+'.pdf')
    end
    
    writematrix(ADF_runtime_means,save_location+'time.xls','WriteMode','overwrite');
    writematrix(ADF_runtime_stds,save_location+'time.xls','WriteMode','append');
    writematrix(PF_runtime_means,save_location+'time.xls','WriteMode','append');
    writematrix(PF_runtime_stds,save_location+'time.xls','WriteMode','append');
    writematrix(discretization_runtime_means,save_location+'time.xls','WriteMode','append');
    writematrix(discretization_runtime_stds,save_location+'time.xls','WriteMode','append');
    writematrix(EP_runtime_means,save_location+'time.xls','WriteMode','append');
    writematrix(EP_runtime_std,save_location+'time.xls','WriteMode','append');
    writematrix(ADF_GMM_runtime_means,save_location+'time.xls','WriteMode','append');
    writematrix(ADF_GMM_runtime_stds,save_location+'time.xls','WriteMode','append');
    
end