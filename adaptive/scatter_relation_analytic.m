function [ m ] = scatter_relation_analytic( c, gc, ns, phi)
%SCATTER_RELATION_ANALYTIC 
% phi is a small angle, theta ~ 1e-5 level.
    INPUT = 1:4; 
    dt = 5e-7;    
    nd = length(phi);
    m = zeros(ns * nd, 4);
    o = zeros(ns * nd, 5);
    s = linspace(0, 2 * pi, ns + 1); s(end)=[];
    
%     d = linspace(0, pi, nd + 2); d(1) = []; d(end) = [];
    loc = [cos(s); sin(s)];
    
    for i = 1:ns
        for j = 1:nd
            theta = s(i) + pi/2 + phi(j);
            v = [cos(theta) sin(theta)] * (1.0/c(loc(1, i), loc(2, i)));
            m((i - 1) * nd + j, INPUT) =  [loc(:, i)' v];
        end
    end
    
    F = @(X)([c(X(1), X(2))^2 * X(3:4)   -(X(3:4)*X(3:4)') * gc(X(1), X(2)) * c(X(1), X(2)) ]);

    for i = 1:ns * nd
        i
        X = m(i, INPUT);
        t = 0; 
        while true
            % RK4            
            k1 = F(X) * dt;
            k2 = F(X + k1/2) * dt;
            k3 = F(X + k2/2) * dt;
            k4 = F(X + k3) * dt;
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6.0;
            t = t + dt;
            if X(1:2) * X(1:2)' >= 1
                break;
            end
        end
        o(i, :) = [X, t];
    end
    
    m = [m o];

end

