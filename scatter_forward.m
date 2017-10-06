%SCATTER_FORWARD forwardly solves the scatter relation on grid, given
%traveltime information.

% input variables:
% m : scatter relation, with Xs and t. [Xs 0 0 0 0 t] is enough.
% N : discretization of a larger domain, default is 1.5^2 square covering
% the disk.
% c : grid based velocity, everywhere in domain, N x N matrix.
%
% output variables:
% s : discretized scatter relation matrix.
% todo: ray information.

function [ s ] = scatter_forward( m, c, ext)
    INPUT = 1:4; OUTPUT = 5:8; TIME = 9; dt = 1e-2;
    N = size(c, 1);
    s = m;s(:, OUTPUT) = 0;
    p = linspace(-ext, ext, N); 
    h = p(2) - p(1);
    
    coef = zeros(N-1, N-1 , 4);
    
    for gi = 1:N-1
        for gj = 1:N-1
            coef(gi, gj, :) = Q4(c, gi, gj, h);
        end
    end
          
    parfor i = 1 : size(m, 1)
        X = m(i, INPUT);
        t = 0;
        while t < m(i, TIME)
            t = t + dt;
            I = floor((X(1) + ext)/ h) + 1;
            J = floor((X(2) + ext)/ h) + 1;
            if (I >= N) 
                I = N - 1;  
            end
            if (J >= N)
                J = N - 1; 
            end

            assert(I >= 1);
            assert(J >= 1);

            k1 = Hamilton(X, coef, p, I, J) * dt;
            k2 = Hamilton(X + k1/2, coef,p, I, J) * dt;
            k3 = Hamilton(X + k2/2, coef, p, I, J) * dt;
            k4 = Hamilton(X + k3, coef, p, I, J) * dt;
            
            X = X + (k1+2*k2+2*k3+k4)/6.0;     
        end
        s(i, OUTPUT) = X;
        
    end
end
