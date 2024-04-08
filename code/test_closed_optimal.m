function test_closed_optimal()
%% test MMD from A. Gretton
alpha = 0.05;
params.numEigs = -1;
N = 1000;
b1_states = normrnd(0.1,2,[N,1]);
b2_states = normrnd(0,2,[N,1]);
b1_probs = normpdf(b1_states,0,2);
b2_probs = normpdf(b2_states,0,2);
b1_probs = b1_probs./sum(b1_probs);
b2_probs = b2_probs./sum(b2_probs);
params.numNullSamp = 100;
params.plotEigs = false;
params.sig = -1;
% b1_sampled_cnts = mnrnd(N,b1_probs);
% b1_resampled_states  = repelem(b1_states,b1_sampled_cnts);
b1_resampled_states = b1_states;
% b2_sampled_cnts = mnrnd(N,b2_probs);
% b2_resampled_states  = repelem(b2_states,b2_sampled_cnts);
b2_resampled_states = b2_states;
tic
[testStat,thresh,params] = mmdTestGamma(b1_resampled_states,b2_resampled_states,alpha,params);
toc
tic
distance = mmd_estimation({b1_probs,b1_states},{b2_probs,b2_states});
toc
%% distribution distance unit test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 1000;
b1_states = normrnd(0,1,[N,1]);
b2_states = normrnd(0,1,[N,1]);
b3_states = normrnd(3,1,[N,1]);
b4_states = normrnd(20,1,[N,1]);
b1_probs = normpdf(b1_states,0,1);
b2_probs = normpdf(b2_states,0,1);
b3_probs = normpdf(b3_states,3,1);
b4_probs = normpdf(b4_states,20,1);
b1_probs = b1_probs./sum(b1_probs);
b2_probs = b2_probs./sum(b2_probs);
b3_probs = b3_probs./sum(b3_probs);
b4_probs = b4_probs./sum(b4_probs);
b1 = {b1_probs,b1_states};
b2 = {b2_probs,b2_states};
b3 = {b3_probs,b3_states};
b4 = {b4_probs,b4_states};
belief_states = {b1,b2,b3,b4};
belief_distances_mmd = zeros([4,4]);
for i=1:4
    bi = belief_states{i};
    for j=1:4
        bj = belief_states{j};
        belief_distances_mmd(i,j) = mmd_estimation(bi,bj); 
    end
end

distances = [];
for i=1:21
    b2_states = normrnd((i-1)*0.5,1,[1000,1]);
    b2_probs = normpdf(b2_states,(i-1)*0.5,1);
    b2_probs = b2_probs./sum(b2_probs);
    b2 = {b2_probs,b2_states};
    distances = [distances,mmd_estimation(b1,b2)]; 
end

X = 0:0.5:10;
figure, plot(X,distances)
xlabel('Mean value of two exact distributions','FontSize', 14)
ylabel('Distance calculated by the MMD','FontSize', 14)
title("Mean values of Gaussians versus the MMD",'FontSize', 14)
%% KNN distance test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 1000;
b1_states = normrnd(0,1,[N,1]);
b1_probs = normpdf(b1_states,0,1);
b1_probs = b1_probs./sum(b1_probs);
b1_distances=abs(repmat(b1_states,[1,N])-b1_states');
b1 = {b1_probs,b1_states,b1_distances};
E = 10;

distances = [];
for m=0:20
    distance = 0;
    for e=1:E
        b2_states = normrnd(m,1,[N,1]);
        b2_probs = normpdf(b2_states,0,1);
        b2_probs = b2_probs./sum(b2_probs);
        b2_distances=abs(repmat(b2_states,[1,N])-b2_states');
        b2 = {b2_probs,b2_states,b2_distances};
        distance=distance+compute_belief_state_distance(b1,b2,N*0.1);
    end
    distances = [distances,distance/E];
end
X=0:20;
figure,
plot(X,distances);
xlabel('Mean value of the belief state','FontSize', 14)
ylabel('Distance','FontSize', 14)
title("Distances between samples from N(0,1) and N(m,1) by KNN",'FontSize', 14)
saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/KNN_distances.pdf');


distances = [];
for m=0:20
    distance = 0;
    for e=1:E
        b2_states = normrnd(m,1,[N,1]);
        b2_probs = normpdf(b2_states,0,1);
        b2_probs = b2_probs./sum(b2_probs);
        b2_distances=abs(repmat(b2_states,[1,N])-b2_states');
        b2 = {b2_probs,b2_states,b2_distances};
        distance=distance+compute_belief_state_distance(b1,b2,0);
    end
    distances = [distances,distance/E];
end
X=0:20;
figure,
plot(X,distances);
xlabel('Mean value of the belief state','FontSize', 14)
ylabel('Distance','FontSize', 14)
title("Distances between samples from N(0,1) and N(m,1) by KDE+JSD",'FontSize', 14)
saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/KDE_JSD_distances.pdf');


% variance

distances = [];
for m=0:20
    distance = 0;
    for e=1:E
        b2_states = normrnd(m,1,[N,1]);
        b2_probs = normpdf(b2_states,0,1);
        b2_probs = b2_probs./sum(b2_probs);
        b2_distances=abs(repmat(b2_states,[1,N])-b2_states');
        b2 = {b2_probs,b2_states,b2_distances};
        distance=distance+compute_belief_state_distance(b1,b2,N*0.1);
    end
    distances = [distances,distance/E];
end
X=0:20;
figure,
plot(X,distances);
xlabel('Mean value of the belief state','FontSize', 14)
ylabel('Distance','FontSize', 14)
title("Distances between samples from N(0,1) and N(0,m) by KNN",'FontSize', 14)
saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/KNN_distances_variance.pdf');


distances = [];

for m=0:20
    distance = 0;
    for e=1:E
        b2_states = normrnd(m,1,[N,1]);
        b2_probs = normpdf(b2_states,0,1);
        b2_probs = b2_probs./sum(b2_probs);
        b2_distances=abs(repmat(b2_states,[1,N])-b2_states');
        b2 = {b2_probs,b2_states,b2_distances};
        distance=distance+compute_belief_state_distance(b1,b2,0);
    end
    distances = [distances,distance/E];
    
end
X=0:20;
figure,
plot(X,distances);
xlabel('Mean value of the belief state','FontSize', 14)
ylabel('Distance','FontSize', 14)
title("Distances between samples from N(0,1) and N(0,m) by KDE+JSD",'FontSize', 14)
saveas(gcf,'experiment_results/greedy ep/online decisions/different_decisions/KDE_JSD_distances_variance.pdf');


%% setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(1);
T = 5;
K = 3;
decision_dim = 1;
decisions = [1:K];

initial_state = 0;
initial_mean = 0; 
initial_var = 1;
initial_model = {initial_mean,initial_var};
transition_models = generate_transition_model(K,3,2,1,1);
H=1.5;
measurement_noise_mu = 1;
measurement_noise_var = 4;
measurement_model = {H,measurement_noise_mu,measurement_noise_var};

prest_decisions = ones([T,1]);
% 
% prest_decisions = [1,2,1,1,1,2];
% 
states_measurements = generate_GMM_states_measurements(initial_state,transition_models,measurement_model,T,prest_decisions);
% states_measurements(:,1) = [2.6386,6.3667,9.4297,12.4952,14.0177,17.6695];
% states_measurements(:,2) = [-0.0044,6.4224,9.5041,14.0385,15.766 9,17.9852];
% 
% X=1:T;
% y1=states_measurements(:,1);
% y2=states_measurements(:,2);
% figure,
% plot(X,y1)
% hold on
% plot(X,y2)
% legend('states','measurements')
% xlabel('Time')
% ylabel('measurements/states')
% figure_title = "states measurements when get stuck";
% title(figure_title)
% hold off
% saveas(gcf,'experiment_results/greedy ep/online decisions/random dynamic transitions/'+figure_title+'.png');
% 
% 
%% MC closed optimal
N = 100;
D = K;
K_nearest = 5;
distance_threshold = 0.5;
epoch_num = 5;
q_table = mc_closed_optimal(N,T,D,K_nearest,initial_state,initial_model,transition_models,measurement_model,distance_threshold,epoch_num);

end