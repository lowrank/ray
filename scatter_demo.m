c = @(x, y)(1 + 0.3 * sin(pi .* x) .* sin(pi .* y));
gc =@(x, y)(0.3 * pi * [cos(pi .* x) .* sin(pi .* y) cos(pi .* y) .* sin(pi .* x) ]);

m = scatter_relation(c, gc, 30, 60);

%%
ext = 1.5; N = 60;
[x,y] = meshgrid(linspace(-ext, ext, N));
c = @(x, y)(1 + 0.4 * sin(pi .* x) .* sin(pi .* y));
dc = c(x, y);

s = scatter_forward(m, dc, ext);
%%
disp((s - m));
