Adaptive phase space method for traveltime in 2D
--
The code computes reconstructs the isotropic wavespeed using multi-arrival time information. The domain is fixed as unit disk.

The code implements the (stabilized) adaptive phase space method. Foliation is guided by fidelity function defined over the physical domain.

To handle the obstacle with cavity, a hybrid method is used.

To run the code for non-obstacle case.

```
julia NonObstacleDemo.jl
```

For obstacle case.

```
julia ObstacleDemo.jl
```