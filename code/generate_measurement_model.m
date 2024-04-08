function measurement_model = generate_measurement_model(dim,rnd_seed)
%     rng(rnd_seed);
%     test for now
    H = diag(ones([1,dim]));
%     H = diag(abs(normrnd(1,0.5,[dim,1])));
    b = zeros([1,dim]);
    m = rand(dim);
    %     test for now
    R = diag(ones([1,dim]));
%     R = diag(abs(normrnd(1,0.5,[dim,1])));
    measurement_model = {H,b,R};
end