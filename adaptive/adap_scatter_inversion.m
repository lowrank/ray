%ADAP_SCATTER_INVERSION adaptively solves the inverse scattering relation
%on grid, traveltime is given.

% input variables:
% m : scatter relation, with Xs and Xr, known.
% ext : 
% c :
% 
% output variables
% s : discretized scattering relation for current guess.
% M : linearization matrix.
function [ s, M ] = adap_scatter_inversion( m, c, ext )
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

    
    

end

