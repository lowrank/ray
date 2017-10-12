function [ M ] = Jacobian(X, coef, grdx, grdy, grxx, grxy, gryy, p, I, J)
    dx = X(1) - p(I);
    dy = X(2) - p(J);
    z = [1, dx, dy, dx*dy];
    lij = coef(I, J, :);
    lc = z * lij(:);

    % better approximation, but slower?
    rij = grdx(I, J, :);
    cij = grdy(I, J, :);
    
    Hxx = grxx(I, J, :);
    Hxy = grxy(I, J, :);
    Hyy = gryy(I, J, :);
    
    lgc = z * [rij(:) cij(:)];
    lh = [z * [Hxx(:) Hxy(:)] ; z * [Hxy(:) Hyy(:)]];


    M = [2 * lc * X(3:4)'*lgc  lc^2 * eye(2); ...
        -(lh * lc  + (lgc'*lgc)) * (X(3:4)*X(3:4)')  -2 * lc * lgc'*X(3:4)];
    


end

