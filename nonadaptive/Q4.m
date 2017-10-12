function [u] = Q4(c, I, J, h)
% Q4 returns Q4 element's coefficients 
    u = zeros(4, 1);
    u(4) = (c(I + 1, J + 1) + c(I, J) - c(I + 1, J) - c(I, J + 1))/h^2;
    u(3) = (c(I, J + 1) - c(I, J))/h; 
    u(2) = (c(I + 1, J) - c(I, J))/h;
    u(1) = c(I, J);
end