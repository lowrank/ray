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

function [ s ] = scatter_forward( m, c, ext)
    INPUT = 1:4; OUTPUT = 5:8; TIME = 9; dt = 1e-2;
    
    N = size(c, 1);
    s = m;s(:, OUTPUT) = 0;
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
    
    for gi = 1:N-1
        for gj = 1:N-1
            coef(gi, gj, :) = Q4(c, gi, gj, h);
        end
    end
    
    for gi = 2:N-2
        for gj = 2:N-2
            grdx(gi, gj, :) = (coef(gi+1, gj, :) - coef(gi-1, gj, :))/(2*h);
            grdy(gi, gj, :) = (coef(gi, gj +1,:) - coef(gi, gj-1,:)) /(2*h);
        end
    end
    
    for gi = 3:N-3
        for gj = 3:N-3
            grxx(gi, gj, :) = (grdx(gi+1, gj, :) - grdx(gi-1, gj,:))/(2*h);
            gryy(gi, gj, :) = (grdy(gi, gj +1,:) - grdy(gi, gj-1,:))/(2*h);
            grxy(gi, gj, :) = (grdx(gi, gj +1,:) - grdx(gi, gj-1,:))/(2*h);
        end
    end
     
    T = m(:, TIME); % avoid broadcasting
    tic;
%     figure;
    parfor i = 1 : size(m, 1)
        X = m(i, INPUT);
        t = 0;
%         ray = [X];
        while t < T(i)
            t = t + dt;
            I = floor((X(1) + ext)/ h) + 1;
            J = floor((X(2) + ext)/ h) + 1;

            k1 = Hamilton(X,        coef, grdx, grdy, p, I, J) * dt;
            k2 = Hamilton(X + k1/2, coef, grdx, grdy, p, I, J) * dt;
            k3 = Hamilton(X + k2/2, coef, grdx, grdy, p, I, J) * dt;
            k4 = Hamilton(X + k3,   coef, grdx, grdy, p, I, J) * dt;
            
            X = X + (k1+2*k2+2*k3+k4)/6.0;     
%             ray = [ray; X];
        end
%         plot(ray(:,1), ray(:,2));
%         hold on;
        s(i, OUTPUT) = X;
    end
    toc;
%     hold off;
end
