%% setting up speed, gradient.
r = @(x,y)(sqrt((x-0.3).^2 + (y-0.3).^2));
v = @(x,y)(sqrt((x+0.2).^2 + (y+0.2).^2));
c = @(x, y)(1 + 1.0 * sin(pi * r(x,y)) + 1.0 * sin(pi*v(x,y)));
gc =@(x, y)(1.0 * pi * cos(pi* r(x,y)) * [(x-0.3)./r(x,y)  (y-0.3)./r(x,y)] + 1.0 * pi * cos(pi * v(x,y)) * [(x+0.2)./v(x,y) (y+0.2)./v(x,y)]  );

%% data generation
ns = 100; na = 100;
m = scatter_relation(c, gc, ns, na);
h = reshape(m(:, 5:8), size(m,1)*4, 1); % make vector of data
%% settings of domain
ext = 2; N = 80;
[y, x] = meshgrid(linspace(-ext, ext, N));
c0 = @(x, y)(0.2 + 0. * sin(pi .* sqrt(x.^2 + y.^2)));
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

reg = (gradx'*gradx + grady'*grady);

%% fidelity term, matches speed dimension.
fid = zeros(N^2, 1);
rnk = zeros(na * ns, 1);
% take hypothesis with 5% level. 
rej_thres = 5e-2; 
alpha = 5e-1;

%% iterative reconstruction
iter = 0;
while true
    [s, ~, M, ti] = scatter_forward(m, c0_, ext, dc);
    q = m(:, 5:8) - s(:, 5:8);
    
    
    % adaptively pick up rays according to distance function.
    % todo: take rank ordering of M.
    for j = 1:na * ns
        rnk(j) = nnz(M(4 * j - 3, :)) - sum(fid(M(4*j-3, :) ~= 0));
    end
    
    [~, p] = sort(rnk); 
%     tv = sum(sqrt(sum(q.^2, 2) ./ sum((m(:, 1:4) - m(:, 5:8)).^2, 2)));
%     ord = find(sqrt(sum(q.^2, 2) ./ sum((m(:, 1:4) - m(:, 5:8)).^2, 2))< 5e-4 * tv);
%     ord = reshape([4 * ord - 3 4 * ord - 2 4 * ord - 1 4 * ord]', size(ord, 1) * 4, 1);

    mx = find(rnk(p) <= 12,1,'last');
    ord = p(1:mx);
    ord = reshape([4 * ord - 3 4 * ord - 2 4 * ord - 1 4 * ord]', size(ord, 1) * 4, 1);

%     ord = 1:4*ns*na;
    
    
    q = reshape(q', ns*na*4, 1);
    
    Mi = M(ord, inn);
     
    tic;
    delta(inn) = (Mi'*Mi + alpha * reg(inn, inn))\(Mi'*q(ord));
    tj = toc;
    
    residue = Mi * delta(inn) - q(ord);
    mis_rate = abs(residue ./ h(ord));
    for k = 1:size(ord, 1)
        if mis_rate(k) < rej_thres
            I = find(M(ord(k), :));
            fid(I) = max(fid(I) , 1 - 10*mis_rate(k));
        end
    end
%     fid(find(sum(M(ord(mis_rate), :), 1))) = 1.0;
    err = norm((reshape(delta, N,N)+c0_-c_).*(sqrt(x.^2 + y.^2) <= 1))/norm(c_.*(sqrt(x.^2 + y.^2) <= 1));
    

    
    c0_ = c0_ + reshape(delta, N, N);
    dc = c(x, y) - c0_;
    
    iter = iter + 1;
    fprintf('%6d \t %6d \t %2.2e \t %2.2e \t %4.4f s \t %4.4f s\n', iter, sum(find(fid)~=0),norm(q)/norm(m(:,5:8)) ,err, ti, tj);
   
    
    subplot(2,2,1);
    surf(x,y, abs(reshape(delta, N,N)+c0_-c_)./c_.*log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0));shading interp;colorbar;view(2);colormap jet;  
    title('relative error');
    subplot(2,2,2);
    surf(x,y, (reshape(fid, N,N)).*log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0), 'EdgeColor', 'none');colorbar;view(2);colormap jet;
    title('auxiliary fidelity');
    subplot(2,2,3);
    surf(x,y, (c_).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0));shading interp;colorbar;view(2);colormap jet;
    title('real speed');
    subplot(2,2,4);
    surf(x,y, ((reshape(delta, N,N)+c0_  ).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0)));shading interp;colorbar;view(2);colormap jet;
    title('recovered speed');  drawnow;
    
    if err < 1e-2 || iter > 25
        break;
    end
end
%%
subplot(2,2,1);
surf(x,y, abs(reshape(delta, N,N)+c0_-c_)./c_.*log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0));shading interp;colorbar;view(2);colormap jet; 
title('relative error');
subplot(2,2,2);
surf(x,y, (reshape(fid, N,N)).*log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0), 'EdgeColor', 'none');colorbar;view(2);colormap jet;
title('auxiliary fidelity');
subplot(2,2,3);
surf(x,y, (c_).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0));shading interp;colorbar;view(2);colormap jet;
title('real speed');
subplot(2,2,4);
surf(x,y, ((reshape(delta, N,N)+c0_  ).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0)));shading interp;colorbar;view(2);colormap jet;
title('recovered speed');  drawnow;

