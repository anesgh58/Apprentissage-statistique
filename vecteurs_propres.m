function [M, U] = vecteurs_propres(data, N)
% data: data from the training database
% N: number of images

M = mean(data,2);

X = (1/sqrt(N))*(data-M);
Gram = X.'*X;

[V, ~] = eig(Gram,'vector');  % Gram est symétrique ->vecteurs propres orthonormés
V = V(:,2:end);

U = X*V*(V.'*(X.')*X*V)^(-1/2);
end