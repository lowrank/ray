%SCATTER_FORWARD forwardly solves the scatter relation on grid, given
%traveltime information.

% input variables:
% m : scatter relation, with Xs and t. [Xs 0 0 0 0 t] is enough.
% ext : domain size, extent could be large enough to tolerate a wrong
% guess.
% c : grid based velocity, everywhere in domain, N x N matrix.
%
% output variables:
% s : discretized scatter relation matrix.
% todo: ray information and fast inversion routine for 4x4 matrix.

function [s, err, M] = scatter_forward( m, c, ext, delta) 
    INPUT = 1:4; OUTPUT = 5:8; TIME = 9; dt = 5e-2;
    
    N = size(c, 1);
    s = m;s(:, OUTPUT) = 0;
    err = zeros(size(m , 1), 4);
    p = linspace(-ext, ext, N); 
    h = p(2) - p(1);
    % 0th, 1th, 2th order derivatives of velocity.
    % gradient and Hessian matrix.
    coef = zeros(N-1, N-1, 4);
    grdx = zeros(N-2, N-2, 4);
    grdy = grdx;
    grxx = zeros(N-3, N-3, 4);
    grxy = grxx;
    gryy = grxx;
    
    df = zeros(N-1, N-1, 4);
    gfx = zeros(N-2, N-2, 4);
    gfy = zeros(N-2, N-2, 4);
    for gi = 1:N-1
        for gj = 1:N-1
            coef(gi, gj, :) = Q4(c, gi, gj, h);
            df  (gi, gj, :) = Q4(delta, gi, gj, h);
        end
    end
    
    for gi = 2:N-2
        for gj = 2:N-2
            grdx(gi, gj, :) = (coef(gi+1, gj, :) - coef(gi-1, gj, :))/(2*h);
            grdy(gi, gj, :) = (coef(gi, gj +1,:) - coef(gi, gj-1,:)) /(2*h);
            gfx (gi, gj, :) = (df(gi+1, gj, :) - df(gi-1, gj,:))/(2*h);
            gfy (gi, gj, :) = (df(gi, gj + 1, :) - df(gi, gj-1,:))/(2*h);
        end
    end
    
    for gi = 3:N-3
        for gj = 3:N-3
            grxx(gi, gj, :) = (grdx(gi+1, gj, :) - grdx(gi-1, gj,:))/(2*h);
            gryy(gi, gj, :) = (grdy(gi, gj +1,:) - grdy(gi, gj-1,:))/(2*h);
            grxy(gi, gj, :) = (grdx(gi, gj +1,:) - grdx(gi, gj-1,:))/(2*h);
        end
    end
     
    dd = reshape(delta, N^2, 1);
    
    T = m(:, TIME); % avoid broadcasting
    tic;
%     figure;
    M = sparse(4 * size(m,1), N^2);
    for i = 1 : size(m, 1)
        X = m(i, INPUT);
        t = 0;
        b = zeros(1, 4);
        Jc = eye(4);
%         ray = [X];
%         A = sparse(4, N^2);
        Ml = sparse(4, N^2);
        while t < T(i)
            t = t + dt;
            I = floor((X(1) + ext)/ h) + 1;
            J = floor((X(2) + ext)/ h) + 1;
            [~, Ap] = Lambda(X, coef, grdx, grdy, df, gfx, gfy, p, I, J);
            Ap = Jc\Ap * dt/2;
            b = b + (Ap * dd)';
            Ml = Ml + Ap;
            k1 = Hamilton(X,        coef, grdx, grdy, p, I, J) * dt;
            k2 = Hamilton(X + k1/2, coef, grdx, grdy, p, I, J) * dt;
            k3 = Hamilton(X + k2/2, coef, grdx, grdy, p, I, J) * dt;
            k4 = Hamilton(X + k3,   coef, grdx, grdy, p, I, J) * dt; 
            
            t1 = Jacobian(X,        coef, grdx, grdy,grxx, grxy, gryy, p, I, J)*Jc * dt;
            t2 = Jacobian(X+k1/2,   coef, grdx, grdy,grxx, grxy, gryy, p, I, J)*(Jc + t1/2) * dt;
            t3 = Jacobian(X+k2/2,   coef, grdx, grdy,grxx, grxy, gryy, p, I, J)*(Jc + t2/2) * dt;
            t4 = Jacobian(X+k3,     coef, grdx, grdy,grxx, grxy, gryy, p, I, J)*(Jc + t3) * dt;
            
            X = X + (k1+2*k2+2*k3+k4)/6.0;     
            Jc = Jc + (t1+2*t2+2*t3+t4)/6.0;

            I = floor((X(1) + ext)/ h) + 1;
            J = floor((X(2) + ext)/ h) + 1;
            [~, Ap] = Lambda(X, coef, grdx, grdy, df, gfx, gfy, p, I, J);
            Ap = Jc\Ap * dt/2;
            b = b + (Ap * dd)';
            Ml = Ml + Ap;
%             ray = [ray; X];
        end
%         plot(ray(:,1), ray(:,2));
%         hold on;
        M(4 * i -3:4*i, :) = Jc * Ml;
        s(i, OUTPUT) = X;
        err(i, :) = ( M(4 * i -3:4*i, :) * dd)';
    end
    toc;
%     hold off;
end
