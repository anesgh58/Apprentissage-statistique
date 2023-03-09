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
    l = length(find(k > alpha));
    U = U(:,N-l:end);

end