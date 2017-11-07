addprocs(3)
using PyPlot
@everywhere using Suppressor
@everywhere @suppress include("NonObstacle.jl")

@everywhere function waveSpeed(x, y)
    r = sqrt((x-0.5)^2 + (y-0.2)^2);
    v = sqrt((x+0.4)^2 + (y+0.3)^2);
    u = sqrt((x-0.3)^2 + (y+0.4)^2);
    w = sqrt((x+0.2)^2 + (y-0.1)^2);
    return 1.0 + 0.2 * sin(pi * r) + 0.4 * sin( pi * v) + 0.2 * sin(pi * u) + 0.3 * sin(pi * w);
end
@everywhere function gradWaveSpeed(x, y)
    r = (sqrt((x-0.5)^2 + (y-0.2)^2));
    v = (sqrt((x+0.4)^2 + (y+0.3)^2));
    u = sqrt((x-0.3)^2 + (y+0.4)^2);
    w = sqrt((x+0.2)^2 + (y-0.1)^2);
    return 0.2 *  pi * cos(pi* r)/r * [(x-0.5), (y-0.2)] +
     0.4 *  pi * cos(pi * v)/v * [(x+0.4), (y+0.3)] +
     0.2 *  pi * cos(pi * u)/u * [(x-0.3), (y+0.4)] +
     0.3 *  pi * cos(pi * w)/w * [(x+0.2), (y-0.1)];
end
################################################################################
numberOfSensor = 50;
numberOfDirect = 100;
timeStep       = 5e-2; # caution small timestep needs more time

################################################################################
@time m = ScatterRelation(waveSpeed, gradWaveSpeed, numberOfSensor,
 numberOfDirect, timeStep, (0,pi));

################################################################################
N = 45; ext = 1.5; # domain
penalty    = 5e-1; # regularization param
rejection  = 5e-2; # fidelity rejection rate
decay      = 10;   # fidelity heristic decay
iteration  = 0;    # iteration number
rankThres  = 12;   # acceptable rays

NonObstacleReconstruction(m, N, ext, penalty, rejection, decay,
     rankThres, waveSpeed);
################################################################################
