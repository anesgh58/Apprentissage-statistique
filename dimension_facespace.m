function [U, l] = dimension_facespace(L, data_trn, U, M, N, alpha, sx, sy, Nc)

F = zeros(sx*length(L),sy*length(L)); %on met beaucoup plus sur les colonnes pour pas que ce soit aplati...
j = 0;
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

        %Question 3 (début)
        if mod(img, 10) == 1 % on va afficher la première image de chaque personne car 10 images par personne
            if l== L(1)
                j = j+1;
            end
            F((i-2)*sx + 1:(i-1)*sx, (j-1)*sy + 1:j*sy) = reconstruire_image(data, U_, M, sx, sy);
        end
       
    end

end

    %Question 3 (fin)
    figure;
    imagesc(F);
    colormap('gray');

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