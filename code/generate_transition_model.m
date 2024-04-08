function [models] = generate_transition_model(D,prior_mu,prior_sigma,num_of_comp,dim,type)
% rng(rnd_seed);

models = {};
% num_of_comp = 2;
if type=="CPOMDP"
    for d = 1:3
        weights_vector = 1;
        coefficients_vector = 1;
        var_vector = 0.05;
        if d==1
            mu_vector = -2;

        elseif d==2
            mu_vector = 2;
        else
            mu_vector = 0;
        end
        model={weights_vector,coefficients_vector,mu_vector,var_vector};
        models{d}=model;
    end
   
else
    
    for d=1:D
        if D==5 && num_of_comp==2
            weights_vector = [0.0822;0.9178];
            coefficients_vector = [2.3150;-0.4327];
            %     mu_vector = [2.3824;1.0423];
            mu_vector = [0;0];
            var_vector = [1.6969;5.1932];
            if d==2
                weights_vector = [0.4079;0.5921];
                coefficients_vector = [1.0344;0.8680];
                %         mu_vector = [1.0900;8.8363];
                mu_vector = [0;0];
                var_vector = [1.7015;2.0477];
            elseif d==3
                weights_vector = [0.6437;0.3563];
                coefficients_vector = [0.1604;0.6479];
                %         mu_vector = [4.9412;-0.5902];
                mu_vector = [0;0];
                var_vector = [0.5917;0.7832];
            elseif d==4
                weights_vector = [0.4592;0.5408];
                coefficients_vector = [1.3900;0.0607];
                %         mu_vector = [2.9483;0.9409];
                mu_vector = [0;0];
                var_vector = [0.2331;4.3131];
            elseif d==5
                weights_vector = [0.9156;0.0844];
                coefficients_vector = [0.3857;0.7425];
                %         mu_vector = [4.1796;2.6234];
                mu_vector = [0;0];
                var_vector = [0.1798;3.9597];
            end
        else
            weights_vector = rand([num_of_comp,1]);
            weights_vector = weights_vector./sum(weights_vector);
            if dim==1
                coefficients_vector = abs(normrnd(1,0.5,[num_of_comp,1]));
                mu_vector = zeros([num_of_comp,1]);
                var_vector = abs(normrnd(prior_mu,prior_sigma,[num_of_comp,1]));
                if d==D
                    coefficients_vector = ones([num_of_comp,1]);
                    var_vector = ones([num_of_comp,1])*0.01;
                end
            else
                mu_vector = zeros([num_of_comp,dim]);
                coefficients_vector = zeros([dim,dim,num_of_comp]);
                for i=1:num_of_comp
                    %             test for now
                    coefficients_vector(:,:,i) = diag(ones([1,dim]));
                    %             coefficients_vector(:,:,i) = diag(abs(normrnd(1,0.5,[dim,1])));
                end
                var_vector = zeros([dim,dim,num_of_comp]);
                for i=1:num_of_comp
                    m = rand(dim);
                    %             test for now
                    var_vector(:,:,i) = diag(ones([1,dim]));
                    %             var_vector(:,:,i) = diag(abs(normrnd(1,0.5,[dim,1])));
                end
            end
        end

    
    %     weights_vector = [0.0822;0.9178];
    %     coefficients_vector = [2.3150;-0.4327];
    % %     mu_vector = [2.3824;1.0423];
    %     mu_vector = [0;0];
    %     var_vector = [1.6969;5.1932];
    %     if k==2
    %         weights_vector = [0.4079;0.5921];
    %         coefficients_vector = [1.0344;0.8680];
    % %         mu_vector = [1.0900;8.8363];
    %         mu_vector = [0;0];
    %         var_vector = [1.7015;2.0477];
    %     elseif k==3
    %         weights_vector = [0.6437;0.3563];
    %         coefficients_vector = [0.1604;0.6479];
    % %         mu_vector = [4.9412;-0.5902];
    %         mu_vector = [0;0];
    %         var_vector = [0.5917;0.7832];
    %     elseif k==4
    %         weights_vector = [0.4592;0.5408];
    %         coefficients_vector = [1.3900;0.0607];
    % %         mu_vector = [2.9483;0.9409];
    %         mu_vector = [0;0];
    %         var_vector = [0.2331;4.3131];
    %     elseif k==5
    %         weights_vector = [0.9156;0.0844];
    %         coefficients_vector = [0.3857;0.7425];
    % %         mu_vector = [4.1796;2.6234];
    %         mu_vector = [0;0];
    %         var_vector = [0.1798;3.9597];
    %     end
    %     weights_vector = [0.44;0.56];
    %     coefficients_vector = [1;1];
    %     mu_vector = [4.6;3.4];
    %     var_vector = [1.5;1.5];
    %     if k==2
    %         weights_vector = [0.65;0.35];
    %         coefficients_vector = [1;1];
    %         mu_vector = [2;6];
    %         var_vector = [1;0.5];
    %     elseif k==3
    %         weights_vector = [0.54;0.46];
    %         coefficients_vector = [1;1];
    %         mu_vector = [4.5;3.5];
    %         var_vector = [0.5;0.5];
    %     elseif k==4
    %         weights_vector = [0.6;0.4];
    %         coefficients_vector = [1;1];
    %         mu_vector = [3.5;4.5];
    %         var_vector = [0.5;0.5];
    %     elseif k==5
    %         weights_vector = [0.55;0.45];
    %         coefficients_vector = [1;1];
    %         mu_vector = [4.6;3.4];
    %         var_vector = [1.5;0.5];        
    %     end
    %     
    %     weights_vector = [1;0];
    %     coefficients_vector = [1;1];
    %     mu_vector = [0;0];
    %     var_vector = [1;1];
    %     if k==2
    %         weights_vector = [0.5;0.5];
    %         coefficients_vector = [1;1];
    %         mu_vector = [0;0];
    %         var_vector = [1;1];
    %     elseif k==3
    %         weights_vector = [0.5;0.5];
    %         coefficients_vector = [1;1];
    %         mu_vector = [0;0];
    %         var_vector = [1;1];
    %     end
        model={weights_vector,coefficients_vector,mu_vector,var_vector};
        models{d}=model;
    end
end


end