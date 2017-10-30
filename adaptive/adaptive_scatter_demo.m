% function adaptive_scatter_demo(coef)
    % generation of data
    r = @(x,y)(sqrt((x-0.2).^2 + (y-0.2).^2));
    v = @(x,y)(sqrt((x+0.3).^2 + (y+0.3).^2));
    
    waveSpeed     = @(x, y)(1 + 0.5 * sin(pi * r(x,y)) + 0.5 * sin(pi*v(x,y)));
    gradWaveSpeed = @(x, y)(0.5 * pi * cos(pi* r(x,y)) * [(x-0.2)./r(x,y)  (y-0.2)./r(x,y)] -...
        0.5 * pi * cos(pi * v(x,y)) * [(x+0.3)./v(x,y) (y+0.3)./v(x,y)]  );
%     r = @(x,y)(sqrt((x-0.5).^2 + (y-0.2).^2));
%     waveSpeed     = @(x, y)(1 + 0.2 * sin(1.5 * pi * r(x,y)));
%     gradWaveSpeed = @(x, y)(0.2 * 1.5 * pi * cos(1.5 * pi* r(x,y)) * [(x-0.5)./r(x,y)  (y-0.2)./r(x,y)]);
    ob = @(x,y)(x^2 + y^2 - 0.25);
    gob = @(x, y)([x ,y]/sqrt(x^2+y^2));

    numberOfSource = 20; 
    numberOfAngle  = 100;
    
    measurements = scatter_relation(waveSpeed, gradWaveSpeed, numberOfSource, numberOfAngle);drawnow;
    target       = reshape(measurements(:, 5:8), size(measurements,1)*4, 1); 
    
    % settings of domain
    ext = 12; N = 240;
    [y, x] = meshgrid(linspace(-ext, ext, N)); 
    dx     = 2 * ext / (N - 1);
    
    % settings of initial guess
%     currentWaveSpeed = @(x, y)(0.2 + 0. * x);
% 
%     c0 = currentWaveSpeed(x, y);
%%
    z = [x(:) y(:)];
    d = size(z, 1);
    m = 5;
    R = zeros(d, (m+1)*(m+2)/2);
    
    col = 0;
    for s = 0:m
        for i = 0:s
            j = s - i;
            col = col + 1;
            % x^i y^j
            for k = 1:d
                R(k, col) = z(k, 1)^i * z(k,2)^j;
            end     
        end
    end

    c0 = reshape(R*coef, N, N);
    %%
    dc = waveSpeed(x,y) - c0;
    c  = c0 + dc;
    

    % All interior index
    ind = (x.^2 + y.^2 <= 1 + 4*dx);
    ind = reshape(ind, N^2, 1);

    % fidelity term, matches speed dimension.
    correction = zeros(N^2, 1);
    reg        = regularization(dx, N);
    fidelity   = zeros(N^2, 1);
    dofs       = zeros(numberOfAngle * numberOfSource, 1);
    rejection  = 5e-2; 
    alpha      = 5e-1;
    decay      = 10.;

    % iterative reconstruction
    iter = 0;
    while true
        [observation, ~, M, cputime] = scatter_forward(measurements, c0, ext, dc);
        mismatch               = measurements(:, 5:8) - observation(:, 5:8);
        
        % get adaptive dofs.
        for j = 1:numberOfAngle * numberOfSource
            dofs(j) = nnz(M(4 * j - 3, :)) - sum(fidelity(M(4 * j - 3, :) ~= 0));
        end
        
        % sort the dofs, getting the permutation.
        [~, perm] = sort(dofs); 
        

        rayInd = find(dofs(perm) <= 20,1,'last');
        order  = perm(1:rayInd);
        order  = reshape([4 * order - 3 4 * order - 2 4 * order - 1 4 * order]', size(order, 1) * 4, 1);

%         order = 1:4*ns*na; % use all information.


        mismatch = reshape(mismatch', numberOfSource*numberOfAngle*4, 1);

        Mi = M(order, ind);

        tic;
        correction(ind) = (Mi'*Mi + alpha * reg(ind, ind))\(Mi'*mismatch(order));
        t = toc;
        % might be good to take some better way to compare vectors.
        residue   = abs(Mi * correction(ind) - mismatch(order));
        
        % setting tentative fidelity using knowledge of mismatch.
        %
        % If the regularization is none, then we should not have such
        % thing, but the discontinuity is prohibited. In other word, the
        % zero-fidelity values basically means there is no ray passing
        % nearby.
        %
        % We put fidelity here is to find out how much error is here by
        % placing a tentative value for picked rays. Since if there is a
        % small error, the mismatch could be large.
        %
        % The fidelity is some value close to rank, but in continuous way,
        % it helps to see how much confidence we can impose on our guess,
        % a known value has fidelity as 1.
        %
        % 
        % there are multiple ways to set this function. 
        % here we take fidelity the same along each ray.
        for k = 1:size(order, 1)
            if residue(k) < rejection
                I = find(M(order(k), :));
                fidelity(I) = max(fidelity(I) , 1 - decay * residue(k));
            end
        end

        err = norm((reshape(correction, N,N)+c0-c).*(sqrt(x.^2 + y.^2) <= 1))/norm(c.*(sqrt(x.^2 + y.^2) <= 1));

        % update all things.

        c0 = c0 + reshape(correction, N, N);
        dc = waveSpeed(x, y) - c0;

        iter = iter + 1;
        
        % output.
        fprintf('%6d \t %6d \t %2.2e \t %2.2e \t %4.4f s \t %4.4f s\n', iter, sum(find(fidelity)~=0),norm(mismatch)/norm(measurements(:,5:8)) ,err, cputime, t);

%%         fh = figure;
        set(gcf,'pos',[200 200 1000 800])
        subplot(2,2,1);
        surf(x,y, abs(reshape(correction, N,N)+c0-c)./c.*log10((sqrt(x.^2 + y.^2) <=1)*10).*log10((sqrt(x.^2 + y.^2) >=0.5)*10));shading interp;colorbar;view(2);colormap jet;  
        title('relative error');
        subplot(2,2,2);
        surf(x,y, (reshape(fidelity, N,N)).*log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0), 'EdgeColor', 'none');colorbar;view(2);colormap jet;
        title('auxiliary fidelity');
        subplot(2,2,3);
        surf(x,y, (c).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0));shading interp;colorbar;view(2);colormap jet;
        title('real speed');
        subplot(2,2,4);
        surf(x,y, ((reshape(correction, N,N)+c0  ).* log10((sqrt(x.^2 + y.^2) <= 1)*10 + (sqrt(x.^2 + y.^2) > 1).*0)));shading interp;colorbar;view(2);colormap jet;
        title('recovered speed');  drawnow;
%%
%         print(fh, '-dpng', '-r0', sprintf('%d.png', iter));
%         close(fh);

        % stop criteria.
        if err < 1e-2 || iter > 50
            break;
        end
    end
% end
