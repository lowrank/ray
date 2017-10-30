% r = @(x,y)(sqrt((x-0.5).^2 + (y-0.2).^2));
% % v = @(x,y)(sqrt((x+0.4).^2 + (y+0.3).^2));
% waveSpeed     = @(x, y)(1 + 0.2 * sin(1.5 * pi * r(x,y)));
% gradWaveSpeed = @(x, y)(0.2 * 1.5 * pi * cos(1.5 * pi* r(x,y)) * [(x-0.5)./r(x,y)  (y-0.2)./r(x,y)]);

r = @(x,y)(sqrt((x-0.2).^2 + (y-0.2).^2));
v = @(x,y)(sqrt((x+0.3).^2 + (y+0.3).^2));

waveSpeed     = @(x, y)(1 + 0.5 * sin(pi * r(x,y)) + 0.5 * sin(pi*v(x,y)));
gradWaveSpeed = @(x, y)(0.5 * pi * cos(pi* r(x,y)) * [(x-0.2)./r(x,y)  (y-0.2)./r(x,y)] -...
    0.5 * pi * cos(pi * v(x,y)) * [(x+0.3)./v(x,y) (y+0.3)./v(x,y)]  );
numberOfSource = 100; 

phi = [1e-4, 2e-4];
nd = length(phi);

measurements = scatter_relation_analytic(waveSpeed, gradWaveSpeed, numberOfSource, phi);
% target       = reshape(measurements(:, 5:8), size(measurements,1)*4, 1); 

mismatch   = measurements(:, 5:8) - measurements(:, 1:4);
traveltime = measurements(:, end);

c = zeros(numberOfSource, 1); % is known from boundary data.
gc = zeros(numberOfSource, 2); % calculated from slowness difference.
% hes = zeros(numberOfSource, 4); % calculated from second order information.

for i = 1:numberOfSource
    c(i) = waveSpeed(measurements((i - 1)*nd + 1, 1), measurements((i - 1)*nd + 1, 2));
    % checking dominating direction, avoid degeneration.
%     [~,ind] = max(measurements((i -1)*nd+1 , 3:4)); 
%     diff = mismatch((i-1)*nd + 1, 3:4);
    gc(i, :) = (-traveltime((i-1)*nd+2) * mismatch((i-1)*nd + 1, 3:4)/traveltime((i-1)*nd+1) * c(i) + traveltime((i-1)*nd+1) * mismatch((i-1)*nd + 2, 3:4)/traveltime((i-1)*nd+2) * c(i))/(traveltime((i-1)*nd+2) - traveltime((i-1)*nd+1) );
end

%% get a polynomial to work.
m = 5;
M = zeros(numberOfSource, (m+1)*(m+2)/2);
N = M;
L = M;
col = 0;
for s = 0:m
    for i = 0:s
        j = s - i;
        col = col + 1;
        % x^i y^j
        for k = 1:numberOfSource
            x = measurements((k - 1)*nd + 1, 1);
            y = measurements((k - 1)*nd + 1, 2);
            M(k, col) = x^i * y^j;
            if i == 0
                N(k, col) = 0;
            else
                N(k, col) = i * x^(i-1) * y^j;
            end
            if j == 0
                L(k, col) = 0;
            else
                L(k, col)  = j * x^i * y^(j-1);
            end
        end     
    end
end
%%
coef = [M;N;L] \ [c; gc(:,1); gc(:,2)];
% coef = M\c;

norm([M;N;L] * coef - [c; gc(:,1); gc(:,2)])

[y, x] = meshgrid(-1:1/90:1);
z = [x(:) y(:)];
d = size(z, 1);

R = zeros(d, (m+1)*(m+2)/2);
col = 0;
for s = 0:m
    for i = 0:s
        j = s - i;
        col = col + 1;
        % x^i y^j
        for k = 1:d
            R(k, col) = z(k,1)^i * z(k,2)^j;
        end     
    end
end

t = R*coef;
e = zeros(d, 1);
for  i = 1:d
    e(i) = waveSpeed(x(i), y(i)) - t(i);
end
%%
th = linspace(0, 2*pi, 2001); th = th(1:end-1);
figure;
hold on;
plot3(cos(th), sin(th), 2*ones(size(th, 2), 1), '-w');
th = linspace(0, 2*pi, 11); th = th(1:end-1);
scatter3(cos(th), sin(th), 2*ones(size(th, 2), 1), 'MarkerEdgeColor','w');
surf(x, y, reshape(e, size(x, 1), size(x,2)), 'edgeColor','none');shading interp;colormap jet;colorbar;view(2);
hold off;

















