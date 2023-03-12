function [true_label] = modifier_labels(lb_test, cls_trn)
    mask = zeros(size(lb_test));
    
    for c = 1:length(cls_trn)
        mask = mask + (lb_test == cls_trn(c)); % on obtient vecteurs avec des 1 la ou e label de la base de test est present dans la base de traininig et de 0 sinon
    end

    true_label = lb_test .* mask;
    true_label = true_label + (true_label == 0).*(max(cls_trn) + 1); %on affecte le prochain label disponible pour suivre la politique de gaussien_classifieur3
end