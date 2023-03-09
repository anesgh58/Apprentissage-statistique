function [] = affichage_nuages(W, mu, dim1, dim2, dim3, dim4, dim5)
[x, ~] = size(W);
affichage_nuages_aux(W, mu, x, dim1, dim2);
affichage_nuages_aux(W, mu, x, dim2, dim3);
affichage_nuages_aux(W, mu, x, dim3, dim4);
affichage_nuages_aux(W, mu, x, dim4, dim5);
end

function [] = affichage_nuages_aux(W, mu, x, dim1, dim2)
% on va tout coder en dur
assert(dim1 > 0);
assert(dim2 > 0);

figure;
hold on;
plot(W(x - dim1 + 1, 1:10), W(x - dim2 + 1, 1:10), "r*");
plot(W(x - dim1 + 1, 11:20), W(x - dim2 + 1, 11:20), "b*");
plot(W(x - dim1 + 1, 21:30), W(x - dim2 + 1, 21:30), "g*");
plot(mu(x - dim1 + 1, 1:3), mu(x - dim2 + 1, 1:3), "k*");
xlabel(strcat('Dimension ', num2str(dim1)));
ylabel(strcat('Dimension ', num2str(dim2)));
legend('personne 1', 'personne 2', 'personne 3', 'moyenne');
grid on;

end