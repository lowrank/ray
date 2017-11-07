addprocs(3)
using PyPlot
@everywhere using Suppressor
@everywhere @suppress include("NonObstacle.jl")
@everywhere @suppress include("Utility.jl")
@everywhere @suppress include("Obstacle.jl")

@everywhere function waveSpeed(x, y)
    r = sqrt((x-0.5)^2 + (y-0.2)^2);
    v = sqrt((x+0.4)^2 + (y+0.3)^2);
    return 1 + 0.4 * sin(pi * r) + 0.4 * sin(pi * v);
end
@everywhere function gradWaveSpeed(x, y)
    r = (sqrt((x-0.5)^2 + (y-0.2)^2));
    v = (sqrt((x+0.4)^2 + (y+0.3)^2));
    return 0.4 * pi * cos(pi* r)/r * [(x-0.5), (y-0.2)] + 0.4 * pi * cos(pi * v)/v * [(x+0.4), (y+0.3)]
end
@everywhere function obstacle(x, y)
    θ = atan2(x,y);
    r = sqrt(x^2 + y^2);
    return r - (0.25 + 0.05 * sin(3 * θ));
end
@everywhere function gradObstacle(x, y)
    θ =  atan2(x,y);
    r =  sqrt(x^2 + y^2);
    n =  [x, y]/r + 0.15 * cos(3 * θ)/r * [-y, x]/r;
    return n/norm(n);
end

################################################################################

T = Dict();tic();
numberOfSensor = 100; # number of sensors placed on boundary.
numberOfDirect = 300; # number of rays emitted, more rays are needed for obstacle case.
timeStep       = 5e-2; # caution small timestep needs more time
m = ScatterRelationObstacle(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
 numberOfDirect, timeStep)

# #filter all rays: orthogonally hitting.
# rayIndex =filter(I-T = Dict();tic();>similarity(m[I,1:2], m[I,5:6]) > 0.995 &&
#  similarity(m[I,3:4],m[I,7:8]) <-0.995, 1:size(m,1))

unbrokenRays = [];
for sIdx = 1:numberOfSensor
    arg = atan2(m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 6],
     m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 5]);
    arg = alignment(arg); # alignment to remove false jumps.
    (lo, hi) = derivativeCheck(arg);
    append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + 1:(sIdx - 1) * numberOfDirect + lo));
    append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + hi:(sIdx) * numberOfDirect));
end

# sIdx = 12
# unbrokenRays = [];
# arg = atan2(m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 6],
#  m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 5]);
# arg = alignment(arg); # alignment to remove false jumps.
# plot(arg)
# n=length(arg);
# q=(arg[2:n]-arg[1:n-1])
# q[13]*q[14]
# q[13]
# q[14]
# plot(q)
# (lo, hi) = derivativeCheck(arg);
# @show(lo,hi)
# append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + 1:(sIdx - 1) * numberOfDirect + lo));
# append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + hi:(sIdx) * numberOfDirect));
# ScatterRelationObstaclePlot(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
#  numberOfDirect, timeStep, unbrokenRays);

m[:, 9] *= timeStep;
m = m[unbrokenRays,:]; # take all unbroken rays.
ScatterRelationObstaclePlot(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
 numberOfDirect, timeStep, unbrokenRays);

target = reshape(m[:,5:8]', 4 * size(m,1), );
################################################################################
T["datagen"] = toq();tic();
# settings
################################################################################
N = 45; ext = 1.5; # domain
penalty    = 5e-1; # regularization param
rejection  = 5e-2; # fidelity rejection rate
decay      = 10;   # fidelity heristic decay
iteration  = 0;    # iteration number
rankThres  = 12;   # acceptable rays
p          = linspace(-ext,ext, N);
hi         = searchsortedfirst(p, 1.0);
lo         = searchsortedlast(p, -1.0);
h          = p[2] - p[1];
c          = zeros(N, N);  # true wave speed
c0         = zeros(N, N);  # recovered wave speed
correction = zeros(N^2);   # correction vector
regularize = regularization(h, N); # regularization matrix.
fidelty    = zeros(N^2);   # fidelity vector.
dofs       = zeros(size(m,1)); # ranks of rays.
################################################################################
T["setting"] = toq();tic();
# setting initial guess
################################################################################
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

    mismatch = reshape(m[:, 5:8]' - observation[:,5:8]', 4 * size(m,1), ); # vector

    # approximation of rank.
    tic();
    Threads.@threads for j = 1:size(m,1)
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
# post-processing.(save, figure, etc.)
################################################################################
NonReflectionPlot(c0, m, ext, timeStep);
pause(20);
