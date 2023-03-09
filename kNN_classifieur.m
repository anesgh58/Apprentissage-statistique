function [est_lb] = kNN_classifieur(data_trn, data_test, lb_trn, U, M, N, N_test, k)

mu = zeros(1,N);
choix = 1:N_test; %numéro de l'image dans la base de test
est_lb = zeros(1,length(choix)); %vecteur contenant le label estimé pour chaque image

for j=1:length(choix) %pour chaque image de la base de test
    data_ = data_test(:,choix(j))-M;
    w = U.'*data_; %vecteur des l composantes principales de l'image de la base de test

    for img_base = 1:N %pour chaque image de la base de training
        data_base = data_trn(:,img_base)-M;
        w_base = U.'*data_base; %vecteur des l composantes principales de l'image de la base de test

        res = norm(w - w_base);
        mu(img_base) = sum(res(:)); %numeros des images de la base de donnees de training les plus ressemblantes
    end

    [m,mu_min] = mink(mu,k);
    phi = lb_trn(mu_min); %labels des correspondant aux numeros d'images

    [count, classes] = hist(phi, unique(phi));
    [maxim,idx] = max(count); %label le plus represente

    est_lb(j)= classes(idx); %label estime de l'image de la base de test
end

end