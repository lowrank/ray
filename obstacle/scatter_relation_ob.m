%SCATTER_RELATION_OB 
function [m] = scatter_relation_ob( c, gc, ns, nd, ob, gob)
    INPUT = 1:4; 
    dt = 1e-2;
    m = zeros(ns * nd, 4);
    o = zeros(ns * nd, 5);
    s = linspace(0, 2 * pi, ns + 1); s(end)=[];
    d = linspace(0, pi, nd + 2); d(1) = []; d(end) = [];
    loc = [cos(s); sin(s)];
    
    for i = 1:ns
        for j = 1:nd
            theta = s(i) + pi/2 + d(j);
            v = [cos(theta) sin(theta)] * (1.0/c(loc(1, i), loc(2, i)));
            m((i - 1) * nd + j, INPUT) =  [loc(:, i)' v];
        end
    end
            
    % driver.
    F = @(X)([c(X(1), X(2))^2 * X(3:4)   -(X(3:4)*X(3:4)') * gc(X(1), X(2)) * c(X(1), X(2)) ]);
    figure;
    for i = 1:ns * nd
        % expand each ray.
        X = m(i, INPUT);
        t = 0;
        res = [X(1:2)'];
        rf = 0;
        while true
            % RK4            
            k1 = F(X) * dt;
            k2 = F(X + k1/2) * dt;
            k3 = F(X + k2/2) * dt;
            k4 = F(X + k3) * dt;
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6.0;
            t = t + dt;
            if X(1:2) * X(1:2)' >= 1
                break; %out.
            end
            
            % if in obstacle or on obstacle, this should be handled
            % carefully. 
            if ob(X(1), X(2)) <= 0 && ~rf
                n = gob(X(1), X(2)); % outward unit col vector
                X(3:4) = X(3:4) * (eye(2) - 2 * (n'*n));
                rf = 1;
            end
            
            res = [res X(1:2)'];
        end
        plot(res(1,:), res(2,:));
        hold on;
        o(i, :) = [X, t];
    end
    hold off;
    m = [m o];
end
