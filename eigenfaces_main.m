% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction - training
% Training set
adr = './database/training3/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end
% Size of the training set
[P,N] = size(data_trn);
% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 
% Display the database
F = zeros(192*Nc,168*max(size_cls_trn));
for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end
figure;
imagesc(F);
colormap(gray);
axis off;

%% Data extraction - test
adr = './database/test1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the images in its columns 
data_test = []; 
lb_test = [];
for i=1:nb_elt
    if fld(i).isdir == false
        lb_test = [lb_test ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_test = [data_test img(:)];
    end
end
[P,N_test] = size(data_test);

%%  Réduction de dimension et eigenfaces
% Question 1
[M, U] = vecteurs_propres(data_trn, N);

% Question2
figure;
for k=1:10
    subplot(10,1,k);
    plot(U(:,k));
end

%Questions 3 et 4
alpha = 0.9;
L = N-1:-1:1;
data = data_trn(:,1);
[U, l] = dimension_facespace(L, data_trn, U, M, N, alpha, 192, 168); % dimension_face contient une fonction d'affichage


%% Travail préalable (pour questions Bonus)
% nous allons modifier le vecteur lb_test pour que les labels nons présents
% dans la base de donnees de training soient remplaces de façon 
% correspondre au label affecte dans gaussien_classifieur3

lb_test = modifier_labels(lb_test, cls_trn);

%% Classifieur k-NN
k = 2; % hyperparametre k
est_lb_knn = kNN_classifieur(data_trn, data_test, lb_trn, U, M, N, N_test, k);

[C_knn,err_rate_knn] = confmat(lb_test,est_lb_knn.');

%% Classifieur gaussien
est_lb_gauss = gaussien_classifieur(data_trn, data_test, lb_trn, cls_trn, U, M, N, N_test, Nc, l);
est_lb_gauss2 = gaussien_classifieur2(data_trn, data_test, lb_trn, cls_trn, U, M, N, N_test, Nc, l);
est_lb_gauss3 = gaussien_classifieur3(data_trn, data_test, lb_trn, cls_trn, U, M, N, N_test, Nc, l, 1e5); %bonus

[C_gauss,err_rate_gauss] = confmat(lb_test, est_lb_gauss.');
[C_gauss2,err_rate_gauss2] = confmat(lb_test, est_lb_gauss2.');
[C_gauss3,err_rate_gauss3] = confmat(lb_test, est_lb_gauss3.'); %bonus

