function [phi] = gaussien_classifieur3(data_trn, data_test, lb_trn, cls_trn, U, M, N, N_test, Nc, l, seuil)
%% Moyenne et Covariance
W = zeros(l,N); % matrice omega contenant les composantes principales de chaque image
for img = 1:N
    data_base = data_trn(:,img)-M;
    W(:,img) = U.'*data_base;
end

%calcul d'une matrice A pour obtenir la moyenne de chaque classe pour
%chaque image avec un simple produit de matrices
[S, LB]= hist(lb_trn,cls_trn); % S pour ne pas avoir a coder en dur combien d'images appartiennent à chaque classe
S = [0 S];
S_ = cumsum(S);
A = zeros(N,S_(Nc+1));

for i=2:length(S)
    A(S_(i-1)+1:S_(i-1)+S(i),S_(i-1)+1:S_(i-1)+S(i))= ones(S(i),S(i))*(1/S(i));
end

Mu = W*A; % moyenne associée à chaque image de la base de training en fonction de sa classe
mu = unique(Mu.', 'rows', 'stable').'; %seulement les mu_j
Sigma = (1/N)*(W-Mu)*(W-Mu).'; % matrice de covariance de toutes les images

%% Classification
W_test = zeros(l,N_test); % matrice omega contenant les composantes principales de chaque image
for img = 1:N_test
    data_base = data_test(:,img)-M;
    W_test(:,img) = U.'*data_base;
end

phi = zeros(1, N_test);
for img = 1:N_test
    res = vecnorm(Sigma^(-0.5) * (W_test(:, img) - mu)).^2;
    
    
    m = mean(res);
    if m > seuil
        [m, idx] = min(res); %argmin des classes
        phi(img) = cls_trn(idx);
    else
        phi(img) = max(cls_trn) + 1; %on lui affecte un label pas encore attribue
    end
end
end