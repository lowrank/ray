c = @(x, y)(1 + 0.3 * sin(pi .* sqrt(x.^2 + y.^2)));
gc =@(x, y)(0.3 * pi * cos(pi .* sqrt(x.^2 + y.^2)) * [x./sqrt(x.^2 + y.^2)  y./sqrt(x.^2 + y.^2)]);

m = scatter_relation(c, gc, 30, 30);

%%
ext = 2.0; N = 60;
[y, x] = meshgrid(linspace(-ext, ext, N));
c1 = @(x, y)(1.0 + 0.0 * sin(pi .* sqrt(x.^2 + y.^2)));
% swap axis
%%
dx = 2 * ext / N;
dc = c1(x, y);
dl = c(x,y) - dc;
dd = dc + dl;
dd = reshape(dd, N^2, 1); % true
%%
inn = (x.^2 + y.^2 <= 1 + 4*dx);
inn = reshape(inn, N^2, 1);
out = (x.^2 + y.^2 > 1 + 4 * dx);
out = reshape(out, N^2, 1);

d = zeros(N^2, 1);
% d(out) = dd(out);

%%

A = sparse(N^2, N^2);
B = sparse(N^2, N^2);
for i = 3:N-3
    for j = 3:N-3
        A(i + (j-1)*N, i + (j - 1)*N) =  1/dx;
        A(i + (j-1)*N, i-1 + (j-1)*N) = -1/dx;
        B(i + (j-1)*N, i + (j-1)*N) = 1/dx;
        B(i + (j-1)*N, i + (j)*N) = -1/dx;
    end
end

REG = (A'*A + B'*B) * dx^2;
%%
for i = 1:10
[s, er, M] = scatter_forward(m, dc, ext, dl);

MO = M(:, out);
MI = M(:, inn);

q = m - s; 
q = q(:, 5:8);
q = reshape(q', 3600, 1);

ql = q;

d(inn) = (MI'*MI + 1e2 * REG(inn, inn))\(MI'*ql);

dd = c(x,y);
disp(norm((reshape(d, N,N)+dc-dd).*(sqrt(x.^2 + y.^2) <= 1))/norm(dd.*(sqrt(x.^2 + y.^2) <= 1)));
surf(x,y, (reshape(d, N,N)+dc));shading interp;view(2);drawnow;

dc = dc + reshape(d, N, N);
dl = c(x, y) - dc;
end


