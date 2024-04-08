function test_GMM_constraint_mm(n_of_components)
%     rng(20221013)
    test_runs = 10;
    numerical_approximation_MIs = [];
    direct_mm_MIs = [];
    constrained_mm_MIs = [];
    jensens_MIs = [];
    numerical_approximation_constrained_MIs = [];
    gmm_weight_collection = [];
    H = 1;
    R = 1.5;
    b = 0;
    colors = [40 116 166;160 64, 0;212 172 13;125 60 152;35 155 86]./255;
    highlight_colors=[0 0 255;255 0 0;255, 255, 0;255, 0, 255;0, 255, 0]./255;
    measurement_model = {H,b,R};
    n = sqrt(n_of_components);
    
    plot_dist_flag = true;
    belief_weights_vector = rand([n,1]);
    belief_weights = belief_weights_vector./sum(belief_weights_vector);
    belief_means_X = normrnd(1,4,[n,1]);
    belief_vars = zeros([n,1]);
    for i=1:n
        var_X = gamrnd(2,2);
        belief_vars(i) = var_X;
    end
    transition_models = generate_transition_model(test_runs,1,3,n,1,"");
    
    for t=1:test_runs
        dynamic_1 = transition_models{t};
         % compute augmented distribution
        gmm_weights = zeros([1,n_of_components]);
        gmm_means = zeros([n_of_components,2]);
        gmm_covs = zeros([2,2,n_of_components]);
        for i=1:n
            
            joint_dist=directly_compute_moment_matching(dynamic_1,measurement_model,belief_means_X(i),belief_vars(i));
            indices = ((i-1)*n+1):(i*n);
            gmm_weights(indices) = joint_dist{1};
            gmm_means(indices,:) = joint_dist{2};
            joint_vars = zeros([2,2,n]);
            for j=1:n
                joint_vars(:,:,j) = reshape(joint_dist{3}(j,:,:),[2,2]);
            end
            gmm_covs(:,:,indices) = joint_vars;
        end
        gmm_weights = gmm_weights./sum(gmm_weights);
        
        [pred_joint_dist,constrained_MI]=gmm_constraint_mm_MI_estimator({belief_weights,belief_means_X,belief_vars},dynamic_1,measurement_model);
        joint_vars = pred_joint_dist{3};
        joint_means = pred_joint_dist{2};
        w_i = pred_joint_dist{1}';
        constrained_mm_MIs = [constrained_mm_MIs,constrained_MI];
        p_gm = gmdistribution(gmm_means,gmm_covs,gmm_weights);
        q_gm = gmdistribution(joint_means,joint_vars,w_i);
        direct_mm_mean = gmm_weights*gmm_means;
        direct_mm_cov = zeros(2);
        for i=1:length(gmm_weights)
            cov = reshape(gmm_covs(:,:,i),[2,2]);
            direct_mm_cov = direct_mm_cov+ gmm_weights(i)*(cov+gmm_means(i,:)'*gmm_means(i,:));
        end
        direct_mm_cov = direct_mm_cov-direct_mm_mean'*direct_mm_mean;
        direct_mm_MI = 0.5*log(direct_mm_cov(1,1)*direct_mm_cov(2,2)/det(direct_mm_cov));
        direct_mm_MIs = [direct_mm_MIs,direct_mm_MI];
        delta_x = 0.1;
        delta_y = 0.1;
        x_min = -18;
        x_max = 18;
        y_min = -18;
        y_max = 18;
        x1 = x_min:delta_x:x_max;
        x2 = y_min:delta_y:y_max;
        if plot_dist_flag
           
            [X1,X2] = meshgrid(x1,x2);
            X = [X1(:) X2(:)];
%             y = mvnpdf(X,direct_mm_mean,direct_mm_cov);
%             y = reshape(y,length(x2),length(x1));
            
            p_gmPDF = arrayfun(@(x,y) pdf(p_gm,[x y]),X(:,1),X(:,2));
            q_gmPDF = arrayfun(@(x,y) pdf(q_gm,[x y]),X(:,1),X(:,2));
    %         for x=1:length(x1)
    %             for y=1:length(x2)
    %                 p_density(y,x) = p_gmPDF(x1(x),x2(y));
    %                 q_density(y,x) = q_gmPDF(x1(x),x2(y));
    %             end
    %         end
            p_density = reshape(p_gmPDF,[length(x2),length(x1)])';
            q_density = reshape(q_gmPDF,[length(x2),length(x1)])';
    %         p_gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(p_gm,[x0 y0]),x,y);
    %         p_density = reshape(p_gmPDF(x1,x2),length(x2),length(x1));
    %         q_gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(q_gm,[x0 y0]),x,y);
    %         q_density = reshape(q_gmPDF(x1,x2),length(x2),length(x1));
    %         contour(x1,x2,y,'--r')
    %         hold on
            figure,
            contour(X1,X2,q_density,10,'-b',"LineWidth",1.5)
            hold on
            contour(X1,X2,p_density,10,'-k',"LineWidth",1.5)
            hold off
            grid on
            legend("Constrained GMM Projection","Original GMM",'FontSize', 12,'Location','northwest')
            xlabel('X','FontSize', 14)
            ylabel('Y','FontSize', 14)
            title("Constrained GMM Projection",'FontSize', 14)
            exportgraphics(gcf,"experiment_results/greedy ep/online decisions/test_cases/approximation_k="+string(n_of_components)+",it="+string(t)+".pdf");
        end
        
        
        

        original_MI_estimation = compute_na_gmm_mi({gmm_weights,gmm_means,gmm_covs},x1,x2);
        numerical_approximation_MIs = [numerical_approximation_MIs,original_MI_estimation];
        constrained_na_MI=compute_na_gmm_mi({w_i,joint_means,joint_vars},x1,x2);
        numerical_approximation_constrained_MIs = [numerical_approximation_constrained_MIs,constrained_na_MI];
        
        determinants = [];
        for i = 1:n_of_components
            determinants = [determinants;det(gmm_covs(:,:,i))];
        end
        original_x_vars = reshape(gmm_covs(1,1,:),[n_of_components,1]);
        original_y_vars = reshape(gmm_covs(2,2,:),[n_of_components,1]);
        jensens_MI = 0.5*gmm_weights*log(original_x_vars)+0.5*gmm_weights*log(original_y_vars)-0.5*gmm_weights*log(determinants);
        jensens_MIs = [jensens_MIs,jensens_MI];

        
    end
%     gmm_weights = [0.2,0.2,0.3,0.3];
%     gmm_means = [1,10;-10,-1;40,10;20,20];
%     gmm_covs = [1,1;1,3];
%     gmm_covs(:,:,2)=[1.5,1;1,2];
%     gmm_covs(:,:,3)=[2,2;2,3];
%     gmm_covs(:,:,4)=[2.5,1;1,1];
    
     X=1:test_runs;
     Y = [numerical_approximation_MIs;direct_mm_MIs;constrained_mm_MIs;numerical_approximation_constrained_MIs;jensens_MIs];
     figure,
     hB = bar(X,Y);
     hT=[]; 
     for i=1:length(hB)  % iterate over number of bar objects
      [y_max,y_index]=max(hB(i).YData);
%       hB(i).FaceColor = 'flat';
%       hsv_data = rgb2hsv(hB(i).CData(y_index,:));
%       hsv_data(2) = 1.0;
%       hB(i).CData(y_index,:) = hsv2rgb(hsv_data);
      hT=[hT,text(hB(i).XData(y_index)+hB(i).XOffset,hB(i).YData(y_index), '\color[rgb]{black} \diamondsuit','VerticalAlignment','bottom','horizontalalign','center')];
     end
     legend("True","Gaussian","Constrained GMM","Numerical GMM","Jensen's",'Location','best','FontSize', 12)
     xlabel('Control Decision','FontSize', 14)
     ylabel('MI estimation','FontSize', 14)
     title("Estimated MI",'FontSize', 14)
%      H = gcf;
%      H.WindowState = 'maximized';
     exportgraphics(gcf,"experiment_results/greedy ep/online decisions/test_cases/MI_estimations_k="+string(n_of_components)+",it="+string(t)+".pdf");
%      saveas(gcf,"experiment_results/greedy ep/online decisions/test_cases/MI_estimations_k="+string(n_of_components)+",it="+string(t)+".png");
     [sorted_na_MIs,sorted_na_idx]=sort(numerical_approximation_MIs,'descend');
     [sorted_mm_MIs,sorted_direct_mm_idx]=sort(direct_mm_MIs,'descend');
     [sorted_con_mm_MIs,sorted_con_mm_idx]=sort(constrained_mm_MIs,'descend');
     disp(sum(sorted_na_idx==sorted_con_mm_idx)/length(sorted_con_mm_idx))
     disp(sum(sorted_na_idx==sorted_direct_mm_idx)/length(sorted_con_mm_idx))
%      [sorted_con_na_MIs,sorted_con_na_idx]=sort(numerical_approximation_constrained_MIs,'descend');
%      [sorted_jensens_MIs,sorted_jensens_idx]=sort(jensens_MIs,'descend');
%      
%      indices = [sorted_na_idx',sorted_direct_mm_idx',sorted_con_mm_idx',sorted_con_na_idx',sorted_jensens_idx'];
%      Y = [sorted_na_MIs;sorted_mm_MIs;sorted_con_mm_MIs;sorted_con_na_MIs;sorted_jensens_MIs];
%      hB=bar(X,Y);
%      hT=[];              % placeholder for text object handles
%     for i=1:length(hB)  % iterate over number of bar objects
%       hT=[hT,text(hB(i).XData+hB(i).XOffset,hB(i).YData, num2str(indices(:,i)),'VerticalAlignment','bottom','horizontalalign','center')];
%     end
%      legend("NA of True","Direct Moment-matching","Constrained MM","NA of Constrained MM","Jensen's inequality ",'Location','northeast','FontSize', 12)
%      xlabel('Order of MI','FontSize', 14)
%      ylabel('MI estimation','FontSize', 14)
%      title("Estimated MI",'FontSize', 14)
%      
%      saveas(gcf,"experiment_results/greedy ep/online decisions/test_cases/Decision_order_k="+string(n_of_components)+",it="+string(t)+".pdf");
end


