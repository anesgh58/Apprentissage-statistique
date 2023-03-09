function [img_ACP] = reconstruire_image(data, U, M, sx, sy)

img_ACP = reshape(U*(U.'*(data-M))+M, [sx,sy]);

% figure;
% imagesc(img_ACP);
% colormap(gray);

end