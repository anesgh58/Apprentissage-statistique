% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training1/';
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

%%  calcul des valeurs propres non nulles et vecteurs propres orthonormés associés de R
% Moyenne empirique
M = mean(data_trn,2);
X = (1/sqrt(N))*(data_trn-M);
Gram = X.'*X;
[V D] = eig(Gram,'vector');  % Gram est symétrique ->vecteurs propres orthonormés
V = V(:,2:end);
U = X*V*(V.'*X.'*X*V)^(-1/2);

% Question2
figure,
for k=1:10
    subplot(10,1,k);
    plot(U(:,k));
end

L = N-1:-1:1;
data = data_trn(:,11);
i = 1;
k = zeros(1, length(L));
for l=L
    U_ = U(:,N-l:end);
    % Question4
    a = norm(U_*(U_.'*(data-M))).^2;
    b =  norm(data - M).^2;
    num =  sum(a(:));
    denum = sum(b(:));
    k(i) = num /denum;
    if(k(i)<0.9)
        l=l+1;
        U = U(:,N-l:end);
        break;
    end
    i=i+1;
    
    % Question3   
    img_ACP = U_*(U_.'*(data-M))+M;
%     figure;
%     imagesc(reshape(img_ACP,[192,168]));
%     colormap(gray);
    
end


%% Classification
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
mu = zeros(1,N);
% choix = randi([1 60],1,1);
k = 2;
choix = 1:N_test;
est_lb = zeros(1,length(choix));
for j=1:length(choix)
    data_ = data_test(:,choix(j))-M;
    w = U.'*data_;
    for img_base = 1:N
        
        data_base = data_trn(:,img_base)-M;
        w_base = U.'*data_base;
        res = norm(w - w_base);
        mu(img_base) = sum(res(:));
        
    end
    
    [m,mu_min] = mink(mu,k);
    % mu_min = mu_min(1);
    % assert(choix == mu_min);
    phi = lb_trn(mu_min);
    % val = unique(phi).';
    % for i=1:length(val)
    %     count(i) = sum(phi == val(i));
    % end
    [count, classes] = hist(phi, unique(phi));
    [maxim,idx] = max(count);
    %classe = find(cls_trn == classes(idx));
    est_lb(j)= classes(idx);
end

[C,err_rate] = confmat(lb_test,est_lb.');

%% Classifieur gaussien
W = zeros(l,N);
for img = 1:N
    data_base = data_trn(:,img)-M;
    W(:,img) = U.'*data_base;
end
[S, LB]= hist(lb_trn,cls_trn);
S = [0 S];
S_ = cumsum(S);
A = zeros(N,S_(Nc+1));

for i=2:length(S)
    A(S_(i-1)+1:S_(i-1)+S(i),S_(i-1)+1:S_(i-1)+S(i))= ones(S(i),S(i))*(1/S(i));
end
Mu = W*A;

Sigma = (1/N)*(W-Mu)*(W-Mu).';




