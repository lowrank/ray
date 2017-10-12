%% setting up speed, gradient.
r = @(x,y)(sqrt((x-0.3).^2 + (y-0.3).^2));
v = @(x,y)(sqrt((x+0.2).^2 + (y+0.2).^2));
c = @(x, y)(1 + 0.4 * sin(pi * r(x,y)) + 0.3 * sin(pi*v(x,y)));
gc =@(x, y)(0.4 * pi * cos(pi* r(x,y)) * [(x-0.3)./r(x,y)  (y-0.3)./r(x,y)] + 0.3 * pi * cos(pi * v(x,y)) * [(x+0.2)./v(x,y) (y+0.2)./v(x,y)]  );

%% data generation
ns = 30; na = 50;
m = scatter_relation(c, gc, ns, na);

%% settings of domain
ext = 2.0; N = 60;
[y, x] = meshgrid(linspace(-ext, ext, N));
c0 = @(x, y)(0.8 + 0. * sin(pi .* sqrt(x.^2 + y.^2)));
dx = 2 * ext / N;
c0_ = c0(x, y);
dc = c(x,y) - c0_;
c_ = c0_ + dc;

% take margin of width of 2 dx.
inn = (x.^2 + y.^2 <= 1 + 4*dx);
inn = reshape(inn, N^2, 1);
out = (x.^2 + y.^2 > 1 + 4 * dx);
out = reshape(out, N^2, 1);

delta = zeros(N^2, 1);

%% regularization
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

reg = (gradx'*gradx + grady'*grady) * dx^2;

%% iterative reconstruction
iter = 0;
while true
    [s, ~, M, ~] = scatter_forward(m, c0_, ext, dc);

    Mi = M(:, inn);
    
    q = m(:, 5:8) - s(:, 5:8);
    q = reshape(q', ns*na*4, 1);

    delta(inn) = (Mi'*Mi + 1e1 * reg(inn, inn))\(Mi'*q);
    
    err = norm((reshape(delta, N,N)+c0_-c_).*(sqrt(x.^2 + y.^2) <= 1))/norm(c_.*(sqrt(x.^2 + y.^2) <= 1));
    surf(x,y, (reshape(delta, N,N)+c0_-c_).*(sqrt(x.^2 + y.^2) <= 1));shading interp;colorbar;view(2);colormap jet;drawnow;

    if err < 1e-2 || iter > 25
        break;
    end
    
    c0_ = c0_ + reshape(delta, N, N);
    dc = c(x, y) - c0_;
    
    iter = iter + 1;
end

