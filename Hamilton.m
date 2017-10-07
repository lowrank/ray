function F = Hamilton(X,coef, grdx, grdy, p, I, J) 
% Hamilton calculates the Hamiltonian system's entries.
    dx = X(1) - p(I);
    dy = X(2) - p(J);
    z = [1, dx, dy, dx*dy];
    lij = coef(I, J, :);
    lc = z * lij(:);
    
    % better approximation, but slower?
    rij = grdx(I, J, :);
    cij = grdy(I, J, :);
    lgc = z * [rij(:) cij(:)];
    
    F = [lc^2 * X(3:4) -lgc * lc * (X(3:4)*X(3:4)')];
end