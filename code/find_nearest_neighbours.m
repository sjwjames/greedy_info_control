function result=find_nearest_neighbours(q_table,t,K,belief_state,distance_threshold)
    if length(q_table)<t
        result = {};
        return
    end
    q_table_entry = q_table{t};
    n = length(q_table_entry);
    distances = zeros([1,n]);
    if n<K
        result = {};
        return
    end
    testStats = zeros([1,n]);
    threshs = zeros([1,n]);
    for i=1:n
        q_table_state = q_table_entry{i}{1};
%         distance=compute_belief_state_distance(belief_state,q_table_state,K);
        [distance,testStat,thresh] = mmd_estimation(belief_state,q_table_state);
        distances(i) = distance;
        testStats(i) = testStat;
        threshs(i) = thresh;
    end
%     disp(distances);
    [k_nearest_distances,indices]=mink(distances,K);
    if testStats(indices(K))>threshs(indices(K))
        result = {};
        return
    end
    k_nearest_beliefs = {};
    k_nearest_q_vals = [];
    for k=1:K
        index = indices(k);
        k_nearest_beliefs{k} = q_table_entry{index}{1};
        k_nearest_q_vals =  [k_nearest_q_vals;q_table_entry{index}{2}];
    end
    result  = {k_nearest_distances,k_nearest_beliefs,k_nearest_q_vals,indices};
    
end

