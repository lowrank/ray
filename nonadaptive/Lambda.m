function [ pV, A] = Lambda( X, coef, grdx, grdy, df, gfx, gfy, p, I, J)
    dx = X(1) - p(I);
    dy = X(2) - p(J);
    h = p(2) - p(1);
    z = [1, dx, dy, dx*dy];
    lij = coef(I, J, :);
    dij = df(I, J,:);
    lc = z * lij(:);
    dc = z * dij(:);

    % better approximation, but slower?
    rij = grdx(I, J, :);
    cij = grdy(I, J, :);
    gfr = gfx(I, J, :);
    gfc = gfy(I, J, :);
    lgc = z * [rij(:) cij(:)];
    gf =  z * [gfr(:) gfc(:)];
    
    pV = [ 2 * lc * dc * X(3:4) (-dc * lgc - lc * gf) * (X(3:4)*X(3:4)')];
    
    % now use row-based scan.
    % dc, gf are discretizing from df (from orginal lambda).
    % it will generate a 4 x N^2 sparse matrix.
    N = size(coef, 1) + 1;
    A = sparse(4, N^2);
    u = dx/h; v = dy/h;
    A(1, I     + (J - 1)*N ) = 2 * lc * X(3) * (1-u) * (1-v);
    A(1, I + 1 + (J - 1)*N ) = 2 * lc * X(3) * u     * (1-v);
    A(1, I     + (J    )*N ) = 2 * lc * X(3) * (1-u) * v;
    A(1, I + 1 + (J    )*N ) = 2 * lc * X(3) * u     * v;
    A(2, I     + (J - 1)*N ) = 2 * lc * X(4) * (1-u) * (1-v);
    A(2, I + 1 + (J - 1)*N ) = 2 * lc * X(4) * u     * (1-v);
    A(2, I     + (J    )*N ) = 2 * lc * X(4) * (1-u) * v;
    A(2, I + 1 + (J    )*N ) = 2 * lc * X(4) * u     * v;
    % thir/fourth  row is more complicated.
    nm = X(3:4)*X(3:4)';
    A(3, I     + (J - 1)*N ) = -nm * lgc(1) * (1-u) * (1-v) + lc * nm * u*(1-v)/(2*h); % ij
    A(3, I + 1 + (J - 1)*N ) = -nm * lgc(1) * u * (1-v) - lc * nm * (1-u)*(1-v)/(2*h) ;
    A(3, I     + (J    )*N ) = -nm * lgc(1) * (1-u) * v + lc * nm * u*v/(2*h);
    A(3, I + 1 + (J    )*N ) = -nm * lgc(1) * u * v - lc * nm * (1-u)*v/(2*h);
    A(3, I + 2 + (J - 1)*N ) = -nm * lc * u * (1-v)/(2*h);
    A(3, I + 2 + (J    )*N ) = -nm * lc * u * v/(2*h);
    A(3, I - 1 + (J - 1)*N ) = nm * lc * (1-u)*(1-v)/(2*h);
    A(3, I - 1 + (J    )*N ) = nm * lc * (1-u)*v/(2*h);
    
    A(4, I     + (J - 1)*N ) = -nm * lgc(2) * (1-u) * (1-v) + lc * nm * (1-u)*v/(2*h); % ij
    A(4, I + 1 + (J - 1)*N ) = -nm * lgc(2) * u * (1-v) + lc * nm * u * v/(2*h);
    A(4, I     + (J    )*N ) = -nm * lgc(2) * (1-u) * v - lc*nm * (1-u)*(1-v)/(2*h);
    A(4, I + 1 + (J    )*N ) = -nm * lgc(2) * u * v - lc * nm * u*(1-v)/(2*h);
    A(4, I     + (J + 1)*N ) = -nm * lc * (1-u) * v/(2*h);
    A(4, I + 1 + (J + 1)*N ) = -nm * lc * u * v/(2*h);
    A(4, I     + (J - 2)*N ) = nm * lc * (1-u)*(1-v)/(2*h);
    A(4, I + 1 + (J - 2)*N ) = nm * lc * u * (1-v)/(2*h);
    
    

end

