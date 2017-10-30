@everywhere @fastmath function Hamilton(phase, c, ∇c)
    speed = c(phase[1], phase[2]);
    H = [speed^2 * phase[3:4]; -(phase[3:4]'*phase[3:4])[1]*∇c(phase[1], phase[2]) * speed];
end

@everywhere @fastmath function DiscreteHamilton(X, eval, grad, p)
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
    return z,I,J, c, gcX, gcY, H
end

@everywhere @fastmath function DiscreteJacobian(X, hess, z, I, J, c, gcX, gcY)
    hXX = (z'* hess[I,J,1:4])[1];
    hXY = (z'* hess[I,J,5:8])[1];
    hYY = (z'* hess[I,J,9:12])[1];
    h = [hXX hXY;hXY hYY]; # hessian
    g = [gcX, gcY];
    τ = (X[3:4]'*X[3:4])[1];
    M = [2 * c * X[3:4] * g' c^2*eye(2); -(c*h + g*g') * τ  -2*c * g*X[3:4]'];
end

@everywhere @fastmath function ScatterRelation(c, ∇c, ns, nd, dt)
    source = linspace(0, 2 * pi, ns + 1); source = source[1:ns];
    direct = linspace(0, pi, nd + 2); direct = direct[2:nd+1];
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
            k1 = Hamilton(X, c, ∇c) * dt;
            k2 = Hamilton(X + k1/2, c, ∇c) * dt;
            k3 = Hamilton(X + k2/2, c, ∇c) * dt;
            k4 = Hamilton(X + k3, c, ∇c) * dt;
            X = X + (k1+2*k2+2*k3+k4)/6.0;
            t = t + dt;
            if norm(X[1:2]) >= 1
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
@everywhere @fastmath function Q4(c, I, J, dx)
    q = [c[I,J], (c[I+1,J]-c[I,J])/dx, (c[I, J+1] - c[I,J])/dx, (c[I+1,J+1]+c[I,J]-c[I+1,J]-c[I,J+1])/dx^2];
end

@everywhere @fastmath function ∂V(X, eval, grad, p)
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
    A = sparse(round(Int64, Rind), round(Int64, Cind), Vind,4, N^2);

end

@everywhere @fastmath function ChunkProcessing!(division, M, s, m, eval, grad, hess, T, p, N, dt)
    idx = indexpids(M)
    if idx == 0
        return;
    end
    for i in division[idx]
        X = m[i, 1:4]; #vector.
        t =  0;
        ρ = eye(4);
        Φ = spzeros(4, N^2); # though sparse.
        @inbounds while (t < T[i]) # one optimization is fixing I,J during each step.
            t = t + dt;
            Θ = inv(ρ) * ∂V(X, eval, grad, p) * dt/2;
            Φ += Θ;
            k1 = DiscreteHamilton(X             , eval, grad, p); # k = z, I,J, c, gcx, gcy, h
            k2 = DiscreteHamilton(X + k1[7]/2*dt, eval, grad, p);
            k3 = DiscreteHamilton(X + k2[7]/2*dt, eval, grad, p);
            k4 = DiscreteHamilton(X + k3[7]*dt  , eval, grad, p);

            v1 = DiscreteJacobian(X                , hess, k1[1], k1[2], k1[3], k1[4], k1[5], k1[6]) * ρ;
            v2 = DiscreteJacobian(X + k1[7]*dt/2   , hess, k2[1], k2[2], k2[3], k2[4], k2[5], k2[6]) * (ρ + v1*dt/2);
            v3 = DiscreteJacobian(X + k2[7]*dt/2   , hess, k3[1], k3[2], k3[3], k3[4], k3[5], k3[6]) * (ρ + v2*dt/2);
            v4 = DiscreteJacobian(X + k3[7]*dt     , hess, k4[1], k4[2], k4[3], k4[4], k4[5], k4[6]) * (ρ + v3*dt);

            X += (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])*dt/6.0;
            ρ += (v1 + 2*v2 + 2*v3 + v4)*dt/6.0;
            Θ =  inv(ρ) * ∂V(X, eval, grad, p) * dt/2;
            Φ += Θ;
        end
        M[(4 * i -3): 4 * (i), :] = ρ*Φ;
        s[i, 5:8] = X;
    end
end

@everywhere @fastmath function ScatterForwardOperator(c, m, ext, dt)
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

@fastmath function regularization(h, N)
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
