function [U, l] = dimension_facespace(L, data_trn, U, M, N, alpha, sx, sy)

for img = 1:N 

    data = data_trn(:, img);
    i = 1;
    k = zeros(1, length(L));

    for l=L

        U_ = U(:,N-l:end);

        %Question 4 (debut)
        num = norm(U_*(U_.'*(data-M))).^2;
        denum =  norm(data - M).^2;
        k(i) = k(i) + num /denum;

        i=i+1;
        
        %Question 3
        if mod(img, 10) == 1 % on va afficher la première image de chaque personne car 10 images par personne
            %figure;
            %imagesc(reconstruire_image(data, U_, M, sx, sy));
            %colormap("gray");
        end
    end

end

    %Question 4 (fin)
    idx = find(k > alpha, 1, "last");
    l = L(idx);
    U = U(:,N-l:end);

    figure;
    plot(L, k);
    hold on;
    plot(alpha * ones(1, length(L)));
    legend('k(l)', 'seuil');
    xlabel("dimension de l'espace de reconstruction");
    ylabel('ratio de reconstruction');
    grid on;

end