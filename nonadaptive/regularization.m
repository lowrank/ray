function [ reg ] = regularization( dx, N )
%REGULARIZATION returns regularization matrix
    xI = zeros(1, 2 * (N - 5)^2);
    xJ = xI;
    xV = xI;
    yI = xI;
    yJ = xI;
    yV = xI;

    row = 1;
    for i = 3:N-3
        for j = 3:N-3
            xI(row) = i + (j - 1)*N;
            xJ(row) = xI(row);
            xV(row) = 1/dx; 
            yI(row) = xI(row);
            yJ(row) = xI(row);
            yV(row) = xV(row);
            row = row + 1;
            xI(row) = i + (j - 1)*N;
            xJ(row) = xI(row) - 1;
            xV(row) = -1/dx; 
            yI(row) = xI(row);
            yJ(row) = xI(row) + N;
            yV(row) = xV(row);
            row = row + 1;
        end
    end

    gradx = sparse(xI, xJ, xV, N^2, N^2);
    grady = sparse(yI, yJ, yV, N^2, N^2);
    reg = (gradx'*gradx + grady'*grady);
end

