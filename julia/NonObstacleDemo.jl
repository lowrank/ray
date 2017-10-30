addprocs(3)

@everywhere include("NonObstacle.jl")

@everywhere function waveSpeed(x, y)
    r = sqrt((x-0.5)^2 + (y-0.2)^2);
    v = sqrt((x+0.4)^2 + (y+0.3)^2);
    return 1 + 0.4 * sin(pi * r) - 0.2 * sin(pi * v)
end
@everywhere function gradWaveSpeed(x, y)
    r = (sqrt((x-0.5)^2 + (y-0.2)^2));
    v = (sqrt((x+0.4)^2 + (y+0.3)^2));
    return 0.4 * pi * cos(pi* r)/r * [(x-0.5), (y-0.2)] - 0.2 * pi * cos(pi * v)/v * [(x+0.4), (y+0.3)]
end
numberOfSensor = 10;
numberOfDirect = 100;
timeStep       = 5e-2;

m = ScatterRelation(waveSpeed, gradWaveSpeed, numberOfSensor,
 numberOfDirect, timeStep)
target = reshape(m[:,5:8]', 4 * numberOfSensor * numberOfDirect, );

N = 60; ext = 2;
p = linspace(-ext,ext, N); h = p[2] - p[1];
c = zeros(N, N);
c0 = zeros(N, N);
correction = zeros(N^2, );
penalty    = 5e-1;
regularize = regularization(h, N);
fidelty    = zeros(N^2, );
dofs       = zeros(numberOfSensor * numberOfDirect , );
rejection  = 5e-2;
decay      = 10;
iteration  = 0;
rankThres  = 12;

Idx = zeros(N^2);
Ldx = zeros(N^2);
for i = 1:N
    for j=1:N
        c[i,j] = waveSpeed(p[i],p[j]);
        if (p[i]^2 + p[j]^2 < 1)
            Ldx[i + (j-1)*N] = 1;
        end
        if p[i]^2 + p[j]^2 <= (1+2*h)^2 # interior
            Idx[i + (j-1)*N] = 1;
            c0[i, j] = c[i,j] - 0.1;
        else
            c0[i, j] = c[i,j]; # fill exterior with known medium.
        end
    end
end
Ldx = find(Ldx);
Idx = find(Idx); gc();

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

    residual = abs(𝔐 * correction[Idx] - mismatch[order]);

    tic();
    for k = 1:size(order, 1)
        if residual[k] < rejection
            id_ = find(M[order[k],:]);
            fidelty[id_] = max(fidelty[id_], 1 - decay * residual[k]);
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
    if iteration >= 20
        break;
    end
    @everywhere gc()
end
using PyPlot
imshow((c0-c));colorbar();show();
