r = @(x,y)(sqrt((x-0.3).^2 + (y-0.3).^2));
v = @(x,y)(sqrt((x+0.2).^2 + (y+0.2).^2));
c = @(x, y)(1 + 0.4 * sin(pi * r(x,y)) + 0.3 * sin(pi*v(x,y)));
gc =@(x, y)(0.4 * pi * cos(pi* r(x,y)) * [(x-0.3)./r(x,y)  (y-0.3)./r(x,y)] + 0.3 * pi * cos(pi * v(x,y)) * [(x+0.2)./v(x,y) (y+0.2)./v(x,y)]  );


ns = 30; na = 50;
m = scatter_relation(c, gc, ns, na);

%%
ext = 2.0; N = 60;
[y, x] = meshgrid(linspace(-ext, ext, N));
c1 = @(x, y)(0.8 + 0. * sin(pi .* sqrt(x.^2 + y.^2)));
% swap axis
%%
dx = 2 * ext / N;
dc = c1(x, y);
dl = c(x,y) - dc;
dd = dc + dl;
% dd = reshape(dd, N^2, 1); % true
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
q = reshape(q', ns*na*4, 1);

ql = q;

d(inn) = (MI'*MI + 1e1 * REG(inn, inn))\(MI'*ql);
dd = c(x,y);
disp(norm((reshape(d, N,N)+dc-dd).*(sqrt(x.^2 + y.^2) <= 1))/norm(dd.*(sqrt(x.^2 + y.^2) <= 1)));
surf(x,y, (reshape(d, N,N)+dc-dd).*(sqrt(x.^2 + y.^2) <= 1));shading interp;colorbar;view(2);colormap jet;drawnow;

dc = dc + reshape(d, N, N);
dl = c(x, y) - dc;
end


