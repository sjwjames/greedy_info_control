function [sssm]=generate_sssm_model(K,D,dim)
    sssm = {};
    r = gamrnd(ones([K,K]),1,K,K);
    rho = r./sum(r,2);
    sssm{1} = rho;
    coefficients = normrnd(1,0.5,[K,D,dim,dim]);
    mu = zeros([K,D,dim,1]);
    cov = gamrnd(2,2,[K,D,dim,dim]);
    cov(:,1) = 0.01;
%     cov(:,2) = 0.02;
%     cov(:,4) = 0.01;
    sssm{2} = {};
    for k=1:K
        for d=1:D
            elem = {};
            elem{1} = reshape(coefficients(k,d,:,:),[dim,dim]);
            elem{2} = reshape(mu(k,d,:,:),[dim,1]);
            elem{3} = reshape(cov(k,d,:,:),[dim,dim]);
            sssm{2}{k}{d}=elem;
        end
    end
end