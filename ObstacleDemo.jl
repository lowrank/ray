addprocs(3)
using PyPlot
@everywhere using Suppressor
@everywhere @suppress include("NonObstacle.jl")
@everywhere @suppress include("Utility.jl")
@everywhere @suppress include("Obstacle.jl")

@everywhere function waveSpeed(x, y)
    # r = sqrt((x-0.5)^2 + (y-0.2)^2);
    # v = sqrt((x+0.4)^2 + (y+0.3)^2);
    # return 1 + 0.4 * sin(pi * r) + 0.4 * sin(pi * v);
    return 1+0.3*sin(1.5*pi*x) * sin(1.5*pi*y);
end
@everywhere function gradWaveSpeed(x, y)
    # r = (sqrt((x-0.5)^2 + (y-0.2)^2));
    # v = (sqrt((x+0.4)^2 + (y+0.3)^2));
    # return 0.4 * pi * cos(pi* r)/r * [(x-0.5), (y-0.2)] + 0.4 * pi * cos(pi * v)/v * [(x+0.4), (y+0.3)]
    return 0.45*pi * [cos(1.5*pi*x)*sin(1.5*pi*y), sin(1.5*pi*x) * cos(1.5*pi*y)];
end
@everywhere function obstacle(x, y)
    θ = atan2.(x,y);
    r = sqrt(x^2 + y^2);
    ρ = 0.2;
    return r - (0.4 + ρ* sin(3 * θ));
end
@everywhere function gradObstacle(x, y)
    θ =  atan2.(x,y);
    r =  sqrt(x^2 + y^2);
    ρ = 0.2;
    n =  [x, y]/r + 3 * ρ * cos(3 * θ)/r * [-y, x]/r;
    return n/norm(n);
end

################################################################################

T = Dict();tic();
numberOfSensor = 50; # number of sensors placed on boundary.
numberOfDirect = 300; # number of rays emitted, more rays are needed for obstacle case.
timeStep       = 5e-2; # caution small timestep needs more time
m = ScatterRelationObstacle(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
 numberOfDirect, timeStep)
################################################################################

#filter all rays: orthogonally hitting.
orthoIndex =filter(I->similarity(m[I,1:2], m[I,5:6]) > 0.995 &&
 similarity(m[I,3:4],m[I,7:8]) <-0.995, 1:size(m,1))

if length(orthoIndex) > 0
    print("reflection detected.\n");
end

unbrokenRays = [];
unbrokenRaysBound=[];



for sIdx =1:numberOfSensor
    arg = atan2(m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 6],
     m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 5]);
    arg = alignment(arg); # alignment to remove false jumps.
    (lo, hi) = derivativeCheck(arg);
    append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + 1:(sIdx - 1) * numberOfDirect + lo));
    append!(unbrokenRays, collect((sIdx - 1) * numberOfDirect + hi:(sIdx) * numberOfDirect));
    append!(unbrokenRaysBound, (sIdx - 1) * numberOfDirect + lo);
    append!(unbrokenRaysBound, (sIdx - 1) * numberOfDirect + hi);
end




sIdx = 30
pp = m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 9].*timeStep;
plot(pp,linewidth=4)
ax = gca()
axis("tight")
ylabel("traveltime")
xlabel("directions")
t = mod(unbrokenRaysBound[(2*sIdx - 1):2 * sIdx],  numberOfDirect)
scatter([t[1],t[2]-1], pp[[t[1], t[2]-1]], s = 1200, alpha=0.4)
ax[:tick_params]("both",labelsize=24)
ax[:set_xlabel]("directions", fontsize=30)
ax[:set_title]("traveltimes of 30th boundary point", fontsize=30)
ax[:set_ylabel]("traveltime", fontsize=30)

pp = m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 5:6].*timeStep;
zz = atan2(pp[:,2],pp[:,1])
n=length(zz)
for i =2:length(zz)
    if abs(zz[i]-zz[i-1]) > 1.5 * pi
        zz[i:n] = zz[i:n] - sign(zz[i] - zz[i-1]) * 2*π;
    end
end

plot(zz,linewidth=4)
ax = gca()
axis("tight")
t = mod(unbrokenRaysBound[(2*sIdx - 1):2 * sIdx],  numberOfDirect)
scatter([t[1],t[2]-1], zz[[t[1], t[2]-1]], s = 1200, alpha=0.4)
ax[:tick_params]("both",labelsize=24)
ax[:set_xlabel]("directions", fontsize=30)
ax[:set_title]("exiting angles of 30th boundary point", fontsize=30)
ax[:set_ylabel]("exiting angles", fontsize=30)

pp = m[(sIdx - 1) * numberOfDirect + 1: sIdx * numberOfDirect, 7:8].*timeStep;
zz = atan2(pp[:,2],pp[:,1])
n=length(zz)
for i =2:length(zz)
    if abs(zz[i]-zz[i-1]) > 1.5 * pi
        zz[i:n] = zz[i:n] - sign(zz[i] - zz[i-1]) * 2*π;
    end
end

plot(zz,linewidth=4)
ax = gca()
axis("tight")
t = mod(unbrokenRaysBound[(2*sIdx - 1):2 * sIdx],  numberOfDirect)
scatter([t[1],t[2]-1], zz[[t[1], t[2]-1]], s = 1200, alpha=0.4)
ax[:tick_params]("both",labelsize=24)
ax[:set_xlabel]("directions", fontsize=30)
ax[:set_title]("exiting locations of 30th boundary point", fontsize=30)
ax[:set_ylabel]("exiting locations", fontsize=30)




m[:, 9] *= timeStep;
s = copy(m); # a copy of data.
m = m[unbrokenRays,:]; # take all unbroken rays.
target = reshape(m[:,5:8]', 4 * size(m,1), );
################################################################################
#=
the plot of non-reflected rays will take a long time, it cannot be parallelized easily.
=#
ScatterRelationObstaclePlot(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
 numberOfDirect, timeStep, orthoIndex);
pause(20);
ScatterRelationObstaclePlot(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
  numberOfDirect, timeStep, unbrokenRaysBound);
pause(20);

################################################################################
T["datagen"] = toq();tic();
# settings
################################################################################
N = 75; ext = 1.5; # domain
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
Edx = zeros(N^2); # effective index. Edx ⊂ Idx. rule out obstacle.
for i = 1:N
    for j=1:N
        c[i,j] = waveSpeed(p[i],p[j]);
        if (obstacle(p[j], p[i]) < 0)
            Edx[i + (j-1)*N] = 1;
        end
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
Edx = find(Edx);
Ldx = find(Ldx);
Idx = find(Idx);

c0 = interpolation(regularize, c0, Ldx, N);
Ldx = setdiff(Ldx, Edx);
mask = NaN*ones(N^2);mask[Ldx] = 0.; mask = reshape(mask, N,N); #NaN mask
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
#=
plot recovered rays.
=#
NonReflectionPlot(c0, s[unbrokenRaysBound,:], ext, timeStep);
pause(30);
################################################################################
#=
use all orthogonal rays.
=#
accurateTimeStep = 5e-3;
m = ScatterRelationObstacle(waveSpeed, gradWaveSpeed, obstacle, gradObstacle, numberOfSensor,
 numberOfDirect, accurateTimeStep); # use finer time step to get accurate information on reflection.

orthoIndex =filter(I->similarity(m[I,1:2], m[I,5:6]) > 0.995 &&
  similarity(m[I,3:4],m[I,7:8]) <-0.995, 1:size(m,1))
m[:,9] *=accurateTimeStep*0.5;
NonReflectionPlot(c0, m[orthoIndex,:], ext, accurateTimeStep);

th = linspace(0, 2*pi, 300);
r = (0.4 + 0.2* sin(4 *th));
xx = r .* cos(th);
yy = r .* sin(th);
plot(yy, xx, "b--");
pause(20);
