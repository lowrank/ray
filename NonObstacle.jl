@everywhere @fastmath function Hamilton(phase::Vector{Float64}, c::Function, ∇c::Function)
    speed = c(phase[1], phase[2]);
    H = [speed^2 * phase[3:4]; -(phase[3:4]'*phase[3:4])[1]*∇c(phase[1], phase[2]) * speed];
end

@everywhere @fastmath function DiscreteHamilton(X::Vector{Float64}, eval::SharedArray{Float64, 3}, grad::SharedArray{Float64, 3},hess::SharedArray{Float64, 3}, p::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})
    N = size(eval, 1) + 1; # recovers the dimension.
    h = p[2] - p[1];
    I = Int64(floor((X[1] - p[1])/ h)) + 1;
    J = Int64(floor((X[2] - p[1])/ h)) + 1;

    dx = X[1] - p[I]; dy = X[2] - p[J];
    u = dx/h; v = dy/h; z = [1,dx ,dy ,dx * dy];
    τ = (X[3:4]'*X[3:4])[1];
    c = (z'*eval[I,J,:])[1];
    gcX = (z'* grad[I,J,1:4])[1];
    gcY = (z'* grad[I,J,5:8])[1];
    H = [c^2 * X[3:4]; -[gcX, gcY] * c * τ];

    hXX = (z'* hess[I,J,1:4])[1];
    hXY = (z'* hess[I,J,5:8])[1];
    hYY = (z'* hess[I,J,9:12])[1];
    h = [hXX hXY;hXY hYY]; # hessian
    g = [gcX, gcY];
    M = [2 * c * X[3:4] * g' c^2*eye(2); -(c*h + g*g') * τ  -2*c * g*X[3:4]'];

    return H, M
end

@everywhere @fastmath function DiscreteHamilton(X::Vector{Float64}, eval::SharedArray{Float64, 3}, grad::SharedArray{Float64, 3}, p::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})
    N = size(eval, 1) + 1; # recovers the dimension.
    h = p[2] - p[1];
    I = Int64(floor((X[1] - p[1])/ h)) + 1;
    J = Int64(floor((X[2] - p[1])/ h)) + 1;

    dx = X[1] - p[I]; dy = X[2] - p[J];
    u = dx/h; v = dy/h; z = [1,dx ,dy ,dx * dy];
    τ = (X[3:4]'*X[3:4])[1];
    c = (z'*eval[I,J,:])[1];
    gcX = (z'* grad[I,J,1:4])[1];
    gcY = (z'* grad[I,J,5:8])[1];
    H = [c^2 * X[3:4]; -[gcX, gcY] * c * τ];

    # hXX = (z'* hess[I,J,1:4])[1];
    # hXY = (z'* hess[I,J,5:8])[1];
    # hYY = (z'* hess[I,J,9:12])[1];
    # h = [hXX hXY;hXY hYY]; # hessian
    # g = [gcX, gcY];
    # M = [2 * c * X[3:4] * g' c^2*eye(2); -(c*h + g*g') * τ  -2*c * g*X[3:4]'];

    return H
end

@everywhere @fastmath function ScatterRelation(c::Function, ∇c::Function, ns::Int, nd::Int, dt::Float64, dir=[0,pi])
    source = linspace(0, 2 * pi, ns + 1); source = source[1:ns];
    direct = linspace(dir[1], dir[2], nd + 2); direct = direct[2:nd+1];
    sensor = [cos.(source) sin.(source)];
    m = SharedArray{Float64}((ns * nd, 9));
    @inbounds @sync @parallel for i = 1:ns
        @inbounds for j = 1:nd
            theta = source[i] + pi * 0.5 + direct[j];
            v = [cos.(theta) sin.(theta)] / (c(sensor[i, 1], sensor[i, 2]));
            m[(i-1)*nd + j, 1:4] = [sensor[i, :] v']; # row vector.
        end
    end

    @inbounds Threads.@threads for i = 1:ns*nd
        X = m[i,1:4];
        t = 0;
        while true
            prev = X;
            k1 = Hamilton(X, c, ∇c) * dt;
            k2 = Hamilton(X + k1/2, c, ∇c) * dt;
            k3 = Hamilton(X + k2/2, c, ∇c) * dt;
            k4 = Hamilton(X + k3, c, ∇c) * dt;
            X = X + (k1+2*k2+2*k3+k4)/6.0;
            t = t + dt;
            if norm(X[1:2]) >= 1
                #=
                requires exact physical coordinate for exits.
                =#
                lo = 0; hi = 1;
                ########################### do while ###########################
                mid = (lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                while abs(e) > 1e-15
                    e > 0? hi = mid:lo=mid;
                    mid =(lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                end
                ################################################################
                t = t - (1-mid)*dt;
                X = 𝔈;
                break;
            end
        end
        m[i, 5:8] = X;
        m[i, 9] = t;
    end
    m
end

# irregular representation.
# I usually refers to y and J refers to x. In following representation, I is x axis, J is y axis.
@everywhere @fastmath function Q4(c::Matrix{Float64}, I::Int, J::Int, dx::Float64)
    q = [c[I,J], (c[I+1,J]-c[I,J])/dx, (c[I, J+1] - c[I,J])/dx, (c[I+1,J+1]+c[I,J]-c[I+1,J]-c[I,J+1])/dx^2];
end

@everywhere @fastmath function ∂V(X::Vector{Float64}, eval::SharedArray{Float64}, grad::SharedArray{Float64}, p::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}})
    N = size(eval, 1) + 1; # recovers the dimension.
    h = p[2] - p[1]; # p[1] = -ext.

    I = Int64(floor((X[1] - p[1])/ h)) + 1;
    J = Int64(floor((X[2] - p[1])/ h)) + 1;

    dx = X[1] - p[I]; dy = X[2] - p[J];
    u = dx/h; v = dy/h; z = [1,dx ,dy ,dx * dy];
    ϕ = [(1-u)*(1-v), u * (1-v), (1-u)*v, u*v];
    ψ = [u*(1-v), -(1-u)*(1-v), u*v, -(1-u)*v];
    γ = [(1-u)*v, u * v, -(1-u)*(1-v), -u*(1-v)];
    τ = (X[3:4]'*X[3:4])[1];
    c = (z'*eval[I,J,:])[1];
    gcX = (z'* grad[I,J,1:4])[1];
    gcY = (z'* grad[I,J,5:8])[1];

    Rind = [ones(4);2 * ones(4); 3 * ones(8); 4 * ones(8)];
    Cind = [I + (J-1)*N, I+1 + (J-1)*N, I + J*N, I+1 + J*N,
            I + (J-1)*N, I+1 + (J-1)*N, I + J*N, I+1 + J*N,
            I + (J-1)*N, I+1 + (J-1)*N, I + J*N, I+1 + J*N,
            I+2 + (J-1)*N,  I-1 + (J-1)*N, I+2 + (J)*N, I-1 + J*N,
            I + (J-1)*N, I+1 + (J-1)*N, I + J*N, I+1 + J*N,
            I + (J+1)*N, I+1 + (J+1)*N, I + (J-2)*N, I+1 + (J-2)*N];
    Vind = [2 * c * X[3] * ϕ;
            2 * c * X[4] * ϕ;
            -τ * gcX * ϕ + c * τ * ψ/(2*h);
            -c * τ * ψ/(2*h);
            -τ * gcY * ϕ + c * τ * γ/(2*h);
            - c* τ * γ/(2*h)];
    A = sparse(round.(Int64, Rind), round.(Int64, Cind), Vind,4, N^2);

end

@everywhere @fastmath function ChunkProcessing!(division::Array{Array{Int, 1},1}, M::SharedArray{Float64, 2}, s::SharedArray{Float64,2}, m, eval::SharedArray{Float64, 3}, grad::SharedArray{Float64, 3}, hess::SharedArray{Float64, 3}, T::Vector{Float64}, p::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, N::Int, timeStep::Float64)
    idx = indexpids(M)
    if idx == 0
        return;
    end
    for i in division[idx]
        X = m[i, 1:4]; #vector.
        t =  0;
        ρ = eye(4);
        Φ = spzeros(4, N^2); # though sparse.
        dt = timeStep;
        @inbounds while (t < T[i]) # one optimization is fixing I,J during each step.
            if ((t + dt) > T[i])
                # reduce the time step at the last step.
                dt = T[i] - t;
            end
            t = t + dt;
            Θ = inv(ρ) * ∂V(X, eval, grad, p) * dt/2;
            Φ += Θ;

            k1, t1 = DiscreteHamilton(X          , eval, grad, hess, p);
            k2, t2 = DiscreteHamilton(X + k1/2*dt, eval, grad, hess, p);
            k3, t3 = DiscreteHamilton(X + k2/2*dt, eval, grad, hess, p);
            k4, t4 = DiscreteHamilton(X + k3*dt  , eval, grad, hess, p);

            v1 = t1 * ρ;
            v2 = t2 * (ρ + v1*dt/2);
            v3 = t3 * (ρ + v2*dt/2);
            v4 = t4*  (ρ + v3*dt);

            X += (k1 + 2*k2 + 2*k3 + k4)*dt/6.0;
            ρ += (v1 + 2*v2 + 2*v3 + v4)*dt/6.0;

            Θ =  inv(ρ) * ∂V(X, eval, grad, p) * dt/2;
            Φ += Θ;
            # when X is outside of extended physical domain, directly cease the ray.
            if (norm(X[1:2]) > 1 + 2*(p[2]-p[1]))
                Θ = inv(ρ) * ∂V(X, eval, grad, p) * (T[i] - t);
                Φ += Θ;
                t += (T[i] - t);
            end

        end
        M[(4 * i -3): 4 * (i), :] = ρ*Φ;
        s[i, 5:8] = X;
    end
end

@everywhere @fastmath function ScatterForwardOperator(c::Matrix{Float64}, m, ext::Float64, dt::Float64)
    N = size(c, 1); num = size(m, 1);
    p = linspace(-ext, ext, N); dx = 2 * ext / (N-1);
    eval = SharedArray{Float64}((N-1, N-1, 4));
    grad = SharedArray{Float64}((N-2, N-2, 8));
    hess = SharedArray{Float64}((N-3, N-3, 12));
    s    = SharedArray{Float64}((num, 9)); # output measurement.
    s[:, 1:4] = m[:, 1:4]; s[:, 9] = m[:, 9];

    # simple loops do not need parallel, extra costs.
    @inbounds for I = 1:N-1
       @inbounds for J=1:N-1
            eval[I,J, :] = Q4(c, I, J, dx);
        end
    end

    @inbounds for I = 2:N-2
       @inbounds for J=2:N-2
            grad[I,J, 1:4] = (eval[I+1,J,:] - eval[I-1,J,:])/(2*dx);
            grad[I,J, 5:8] = (eval[I,J+1,:] - eval[I,J-1,:])/(2*dx);
        end
    end

    @inbounds for I = 3:N-3
       @inbounds for J=3:N-3
            hess[I,J, 1:4]  = (grad[I+1,J,1:4] - grad[I-1,J,1:4])/(2*dx);
            hess[I,J, 5:8]  = (grad[I,J+1,1:4] - grad[I,J-1,1:4])/(2*dx);
            hess[I,J, 9:12] = (grad[I,J+1,5:8] - grad[I,J-1,5:8])/(2*dx);
        end
    end

    T = m[:, 9];
    M = SharedArray{Float64}((4 * num, N^2));

    # balancer
    order = sortperm(T, rev=true);
    nprocs = length(procs(M));
    division = [Int[] for i = 1:nprocs];
    timeSum = zeros(nprocs);

    for i = 1:num
        minInd = indmin(timeSum);
        timeSum[minInd] += T[order[i]];
        push!(division[minInd], i);
    end

    # scheduler
    @sync begin
        for pid in procs(M)
            @async remotecall_wait(ChunkProcessing!, pid, division, M, s, m, eval, grad, hess, T, p, N, dt)
        end
    end

    Base.SparseArrays.droptol!(sparse(M), 1e-12, false),s # compress the matrix
end

@fastmath function regularization(h::Float64, N::Int)
    xI = zeros(Int, 2 * (N-5)^2);
    xJ = zeros(Int, 2 * (N-5)^2);
    xV = zeros(2*(N-5)^2);
    yI = zeros(Int,  2 * (N-5)^2);
    yJ = zeros(Int,  2 * (N-5)^2);
    yV = zeros(2*(N-5)^2);
    row = 1;
    for i = 3:N-3
        for j = 3:N-3
            xI[row] = i + (j - 1)*N;
            xJ[row] = xI[row];
            xV[row] = 1.0/h;
            yI[row] = xI[row];
            yJ[row] = xI[row];
            yV[row] = xV[row];
            row += 1;
            xI[row] = i + (j - 1)*N;
            xJ[row] = xI[row] - 1;
            xV[row] = -1.0/h;
            yI[row] = xI[row];
            yJ[row] = xI[row] + N;
            yV[row] = xV[row];
            row += 1;
        end
    end
    gradientX = sparse(xI, xJ, xV, N^2, N^2);
    gradientY = sparse(yI, yJ, yV, N^2, N^2);
    r = (gradientX'*gradientX + gradientY'*gradientY);
end

@fastmath function interpolation(R, c0::Matrix{Float64}, Idx::Vector{Int}, N::Int)
    z = reshape(c0, N^2);
    b = -R * z;
    z[Idx] = R[Idx, Idx]\b[Idx];
    return reshape(z, N,N);
end

function NonObstacleReconstruction(m::SharedArray{Float64,2}, N::Int, ext::Float64, penalty::Float64, rejection::Float64, decay::Float64,
    rankThres::Int, waveSpeed::Function, timeStep::Float64)
################################################################################
    T = Dict();tic();
    target = reshape(m[:,5:8]', 4 * size(m, 1), );
################################################################################
    iteration  = 0;
    p          = linspace(-ext,ext, N);
    hi         = searchsortedfirst(p, 1.0);
    lo         = searchsortedlast(p, -1.0);
    h          = p[2] - p[1];
    c          = zeros(N, N);  # true wave speed
    c0         = zeros(N, N);  # recovered wave speed
    correction = zeros(N^2);   # correction vector
    regularize = regularization(h, N); # regularization matrix.
    fidelty    = zeros(N^2);   # fidelity vector.
    dofs       = zeros(size(m, 1)); # ranks of rays.
################################################################################
    T["setting"] = toq();tic();
    Idx = zeros(N^2); # interior index
    Ldx = zeros(N^2); # local variable index, Idx ⊂ Ldx.
    for i = 1:N
        for j=1:N
            c[i,j] = waveSpeed(p[i],p[j]);
            if (p[i]^2 + p[j]^2 < 1)
                Ldx[i + (j-1)*N] = 1;
            else
                c0[i, j] = c[i,j]; # fill exterior with known medium.
            end
            if p[i]^2 + p[j]^2 <= (1+2*h)^2 # interior
                Idx[i + (j-1)*N] = 1;
            end
        end
    end
    Ldx = find(Ldx);
    Idx = find(Idx);

    c0 = interpolation(regularize, c0, Ldx, N);
    mask = NaN*ones(N^2);mask[Ldx] = 0.;mask = reshape(mask, N,N); #NaN mask
    @everywhere gc(); # gc before going into main loop.
################################################################################
    cmap = PyPlot.cm[:jet];
    cmap[:set_bad]("white",1.);
    show();ion();fig = figure(figsize=(10,8));
    T["initial"] = toq(); tic();
    # main loop
################################################################################

    while true
        tic();
        M, observation = ScatterForwardOperator(c0, m, ext, timeStep); # time complexity is O(N^2).
        t_forward = toq();

        mismatch = reshape(m[:, 5:8]' - observation[:,5:8]', 4 * numberOfDirect * numberOfSensor, ); # vector

        # approximation of rank.
        tic();
        Threads.@threads for j = 1:numberOfSensor * numberOfDirect
            dofs[j] = nnz(M[4 * j - 3, :]) - sum(fidelty[find(M[4 * j - 3, :])]);
        end
        t_dof = toq();

        perm = sortperm(dofs, rev=false);
        trunc = searchsortedlast(dofs[perm], rankThres);

        order = perm[1:trunc];
        order = reshape([4 * order - 3 4 * order - 2 4 * order - 1 4 * order]', size(order, 1) * 4, );

        𝔐 = M[order, Idx];

        A = (𝔐'*𝔐 + penalty * regularize[Idx,Idx]);
        b = 𝔐'*mismatch[order];
        tic();
        correction[Idx] =   A\b;
        t_solv = toq();

        residual = abs.(𝔐 * correction[Idx] - mismatch[order]);

        tic();
        for k = 1:size(order, 1)
            if residual[k] < rejection
                id_ = find(M[order[k],:]);
                fidelty[id_] = max.(fidelty[id_], 1 - decay * residual[k]);
            end
        end
        t_fid = toq();

        c0 = c0 + reshape(correction, N,N);
        error = norm(reshape(c-c0,N^2,1)[Ldx])/norm(reshape(c,N^2,1)[Ldx]);
        objective = norm(mismatch)/norm(target);
        if iteration == 0
            print(@sprintf("%6s\t%6s\t%10s\t%10s\t%6s\t%6s\t%6s\t%6s\n", "iter", "rank", "obj", "err", "T1", "T2", "T3","T4"));
        end
        print(@sprintf("%6d\t%6.2d\t%10.2e\t%10.2e\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n", iteration, sum(fidelty), objective , error, t_forward, t_dof, t_solv, t_fid));

        iteration += 1;
        if iteration >= 50 || objective < 1e-2 # when scatter relation has been recovered, iteration stops.
            break;
        end

        clf();
        ax = subplot("221");
        ax[:set_title]("error of speed");
        z = c-c0;z += mask;
        imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
        interpolation="bilinear", cmap = cmap);colorbar();
        ax = subplot("222");
        ax[:set_title]("auxiliary fidelity");
        z = reshape(fidelty,N,N);z += mask;
        imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
        interpolation="none", cmap = cmap);colorbar();
        ax = subplot("223");
        ax[:set_title]("true speed");
        z = c;z += mask;
        imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
        interpolation="bilinear", cmap = cmap);colorbar();
        ax = subplot("224");
        ax[:set_title]("recovered speed");
        z = c0;z += mask;
        imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
        interpolation="bilinear", cmap = cmap);colorbar();
        draw();

        @everywhere gc();
    end
    ################################################################################
    T["solving"] = toq();
    for el in T
        print(@sprintf("%8s: %6.2f s\n", el[1], el[2]));
    end
    @everywhere gc();


    clf();
    ax = subplot("221");
    ax[:set_title]("error of speed");
    z = c-c0;z += mask;
    imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
    interpolation="bilinear", cmap = cmap);colorbar();
    ax = subplot("222");
    ax[:set_title]("auxiliary fidelity");
    z = reshape(fidelty,N,N);z += mask;
    imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
    interpolation="none", cmap = cmap);colorbar();
    ax = subplot("223");
    ax[:set_title]("true speed");
    z = c;z += mask;
    imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
    interpolation="bilinear", cmap = cmap);colorbar();
    ax = subplot("224");
    ax[:set_title]("recovered speed");
    z = c0;z += mask;
    imshow(z[lo:hi, lo:hi], extent = [p[lo], p[hi],p[lo],p[hi]],
    interpolation="bilinear", cmap = cmap);colorbar();
    draw();
    @show();


end
