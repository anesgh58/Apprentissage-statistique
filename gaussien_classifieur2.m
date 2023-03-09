function [phi] = gaussien_classifieur(data_trn, data_test, lb_trn, cls_trn, U, M, N, N_test, Nc, l)
%% Moyenne et Covariance
W = zeros(l,N); % matrice omega contenant les composantes principales de chaque image
for img = 1:N
    data_base = data_trn(:,img)-M;
    W(:,img) = U.'*data_base;
end

[S, LB]= hist(lb_trn,cls_trn); 
S = [0 S];
S_ = cumsum(S);

Mu = zeros(l, Nc);
Sigma = zeros(l*l, Nc);
for i=2:length(S)
    W_ = W(:, S_(i-1) + 1: S_(i-1) + S(i));
    mu = sum(W_, 2)/S(i);
    Mu(:, i-1) = mu; % vecteur contenat sur chaque colonne la moyenne de chaque classe
    Sigma(:, i-1) = reshape((1/S(i)) * (W_ - mu) * (W_ - mu).', [l*l, 1]); %vecteur contenat la version vectorisee de la matrice de covariance de chaque classe

end

%% Classification
W_test = zeros(l,N_test); % matrice omega contenant les composantes principales de chaque image
for img = 1:N_test
    data_base = data_test(:,img)-M;
    W_test(:,img) = U.'*data_base;
end

phi = zeros(1, N_test);
for img = 1:N_test
    res = zeros(1, Nc);
    CP = W_test(:, img);

    for x = 1:Nc
        res(x) = norm(reshape(Sigma(:, x), [l, l])^(-0.5) * (CP - Mu(:, x)))^2;
    end

    [m, idx] = min(res); %argmin des classes
    phi(img) = cls_trn(idx);
end

end