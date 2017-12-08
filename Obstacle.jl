#===============================================================================
The inputs:
- c, ∇c are functions.
- ob is zero-levelset of interface, ∇ob is normal direction.
- ns, nd are ray-dependent parameters.
- dt is time step for simulation.
===============================================================================#
# @everywhere @fastmath function ScatterRelationObstacleUntrim(m, c, ∇c, ob, ∇ob, dt)
#
# end
@everywhere @fastmath function ScatterRelationObstacle(c, ∇c, ob, ∇ob, ns, nd, dt)
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
    @inbounds for i = 1:ns*nd
        X = m[i,1:4];
        t = 0;
        rfl = 0;
        while true
            if (ob(X[1], X[2]) > 0 )
                rfl = 0;
            end

            prev = X; # before hitting interface.

            k1 = Hamilton(X, c, ∇c) * dt;
            k2 = Hamilton(X + k1/2, c, ∇c) * dt;
            k3 = Hamilton(X + k2/2, c, ∇c) * dt;
            k4 = Hamilton(X + k3, c, ∇c) * dt;
            X = X + (k1+2*k2+2*k3+k4)/6.0;
            t = t + 1;
            if norm(X[1:2]) >= 1
                #=
                requires exact exiting physical location.
                Again, using information of prev and current X.
                =#
                lo = 0; hi = 1;
                ########################### do while ###########################
                mid = (lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                while abs(e) > 1e-15
                    e > 0? hi = mid:lo=mid;
                    mid =(lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                end
                ################################################################
                t = t - (1-mid);
                # res = [res 𝔈[1:2]];
                X = 𝔈;
                break;
            end
            #= add reflection here.
            How to precisely describe the reflection point in solving ODE.
            - linear interpolation.
                X_s = αX_p + (1-α)X_n ∈ ∂D.
                t_s = αt_p + (1-α)t_n
                second error expected in time.
            - extrapolation.
                from X_p to X_n, it is nonlinear, but short time. One can use
                short-time expansion to solve a quadratic equation, get third '
                order error.
            =#
            if (ob(X[1], X[2]) <= 0 && rfl == 0)
                # reflection does not happen.
                # prev and X are separated by ob.
                # mid is the hitting time. T_hit = t + dt * mid;
                # memorize this hitting time?
                lo = 0; hi = 1;rfl=1;
                ########################## do while ############################
                mid=(lo + hi)/2;𝕽=(1-mid) * prev + mid * X; e = ob(𝕽[1], 𝕽[2]);
                while abs(e) > 1e-15 # tolerance, depending on the ob function.
                    e > 0 ? lo=mid:hi=mid;
                    mid=(lo + hi)/2;𝕽=(1-mid)*prev + mid*X; e = ob(𝕽[1], 𝕽[2]);
                end
                ################################################################
                # res = [res 𝕽[1:2]]; # reflection
                # reverse it! Trace back for a short time dt * (1 - mid);
                n = gradObstacle(𝕽[1], 𝕽[2]);
                X[1:2] = 𝕽[1:2];X[3:4] = (eye(2) - 2 * n*n')*𝕽[3:4];
                # k1 = Hamilton(X, c, ∇c) * dt * (1-mid);
                # k2 = Hamilton(X + k1/2, c, ∇c) * dt * (1-mid);
                # k3 = Hamilton(X + k2/2, c, ∇c) * dt * (1-mid);
                # k4 = Hamilton(X + k3, c, ∇c) * dt * (1-mid);
                # X = X + (k1+2*k2+2*k3+k4)/6.0;
            end
        end
        m[i, 5:8] = X;
        m[i, 9] = t;
    end
    m
end
@everywhere @fastmath function ScatterRelationObstaclePlot(c, ∇c, ob, ∇ob, ns, nd, dt, rayIndex)
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

    @inbounds for i in rayIndex
        X = m[i,1:4];
        t = 0;
        rfl = 0;
        res = X[1:2];
        while true
            if (ob(X[1], X[2]) > 0 )
                rfl = 0;
            end
            prev = X; # before hitting interface.

            k1 = Hamilton(X, c, ∇c) * dt;
            k2 = Hamilton(X + k1/2, c, ∇c) * dt;
            k3 = Hamilton(X + k2/2, c, ∇c) * dt;
            k4 = Hamilton(X + k3, c, ∇c) * dt;
            X = X + (k1+2*k2+2*k3+k4)/6.0;
            t = t + 1;
            if norm(X[1:2]) >= 1
                #=
                requires exact exiting physical location.
                Again, using information of prev and current X.
                =#
                lo = 0; hi = 1;
                ########################### do while ###########################
                mid = (lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                while abs(e) > 1e-15
                    e > 0? hi = mid:lo=mid;
                    mid =(lo+hi)/2;𝔈 =(1-mid)*prev+mid*X; e = norm(𝔈[1:2]) - 1;
                end
                ################################################################
                t = t - (1-mid);
                res = [res 𝔈[1:2]];
                X = 𝔈;
                break;
            end

            #= add reflection here.
            How to precisely describe the reflection point in solving ODE.
            - linear interpolation.
                X_s = αX_p + (1-α)X_n ∈ ∂D.
                t_s = αt_p + (1-α)t_n
                second error expected in time.
            - extrapolation.
                from X_p to X_n, it is nonlinear, but short time. One can use
                short-time expansion to solve a quadratic equation, get third '
                order error.
            =#
            if (ob(X[1], X[2])  <= 0 && rfl == 0)
                # reflection does not happen.
                # prev and X are separated by ob.
                # mid is the hitting time. T_hit = t + dt * mid;
                # memorize this hitting time?
                lo = 0; hi = 1;rfl=1;
                ########################## do while ############################
                mid=(lo + hi)/2;𝕽=(1-mid) * prev + mid * X; e = ob(𝕽[1], 𝕽[2]);
                while abs(e) > 1e-15 # tolerance, depending on the ob function.
                    e > 0 ? lo=mid:hi=mid;
                    mid=(lo + hi)/2;𝕽=(1-mid)*prev + mid*X; e = ob(𝕽[1], 𝕽[2]);
                end
                ################################################################
                res = [res 𝕽[1:2]]; # reflection
                # reverse it! Trace back for a short time dt * (1 - mid);
                n = gradObstacle(𝕽[1], 𝕽[2]);
                X[1:2] = 𝕽[1:2];X[3:4] = (eye(2) - 2 * n*n')*𝕽[3:4];
                # k1 = Hamilton(X, c, ∇c) * dt * (1-mid);
                # k2 = Hamilton(X + k1/2, c, ∇c) * dt * (1-mid);
                # k3 = Hamilton(X + k2/2, c, ∇c) * dt * (1-mid);
                # k4 = Hamilton(X + k3, c, ∇c) * dt * (1-mid);
                # X = X + (k1+2*k2+2*k3+k4)/6.0;

            end



            # if (ob(X[1], X[2]) <= 0 && rfl == 1)
            #     # reflection does not happen.
            #     # prev and X are separated by ob.
            #     # mid is the hitting time. T_hit = t + dt * mid;
            #     # memorize this hitting time?
            #     lo = 0; hi = 1; rfl = 2;
            #     ########################## do while ############################
            #     mid=(lo + hi)/2;𝕽=(1-mid) * prev + mid * X; e = ob(𝕽[1], 𝕽[2]);
            #     while abs(e) > 1e-15 # tolerance, depending on the ob function.
            #         e > 0 ? lo=mid:hi=mid;
            #         mid=(lo + hi)/2;𝕽=(1-mid)*prev + mid*X; e = ob(𝕽[1], 𝕽[2]);
            #     end
            #     ################################################################
            #     res = [res 𝕽[1:2]]; # reflection
            #     # reverse it! Trace back for a short time dt * (1 - mid);
            #     n = gradObstacle(𝕽[1], 𝕽[2]);
            #     X[1:2] = 𝕽[1:2];X[3:4] = (eye(2) - 2 * n*n')*𝕽[3:4];
            #     k1 = Hamilton(X, c, ∇c) * dt * (1-mid);
            #     k2 = Hamilton(X + k1/2, c, ∇c) * dt * (1-mid);
            #     k3 = Hamilton(X + k2/2, c, ∇c) * dt * (1-mid);
            #     k4 = Hamilton(X + k3, c, ∇c) * dt * (1-mid);
            #     X = X + (k1+2*k2+2*k3+k4)/6.0;
            # end
            res = [res X[1:2]];
        end
        m[i, 5:8] = X;
        m[i, 9] = t;
        #=
        =#
        plot(res[1,:], res[2,:]);
        xlabel("x");ylabel("y");
        title("selected rays");
        axes()[:set_aspect]("equal","datalim");
    end
    show();
    m
end
@everywhere @fastmath function NonReflectionPlot(c, m, ext, dt)
    N = size(c, 1); num = size(m, 1);
    p = linspace(-ext, ext, N); dx = 2 * ext / (N-1);
    eval = SharedArray{Float64}((N-1, N-1, 4));
    grad = SharedArray{Float64}((N-2, N-2, 8));
    hess = SharedArray{Float64}((N-3, N-3, 12));

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

    for i =1:num
        X = m[i, 1:4]; #vector.
        t =  0;
        res = X[1:2];
        @inbounds while (t < T[i]) # one optimization is fixing I,J during each step.
            t = t + dt;
            k1 = DiscreteHamilton(X             , eval, grad, p); # k = z, I,J, c, gcx, gcy, h
            k2 = DiscreteHamilton(X + k1[7]/2*dt, eval, grad, p);
            k3 = DiscreteHamilton(X + k2[7]/2*dt, eval, grad, p);
            k4 = DiscreteHamilton(X + k3[7]*dt  , eval, grad, p);

            X += (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])*dt/6.0;
            # when X is outside of extended physical domain, directly cease the ray.
            res = [res X[1:2]]
        end
        plot(res[1,:], res[2,:]);
        xlabel("x");ylabel("y");
        title("non-reflected rays");
        axes()[:set_aspect]("equal","datalim");
    end
    show();
end
@everywhere @fastmath function NonReflectionTrace(c, m, ext, dt)
    N = size(c, 1); num = size(m, 1);
    p = linspace(-ext, ext, N); dx = 2 * ext / (N-1);
    eval = SharedArray{Float64}((N-1, N-1, 4));
    grad = SharedArray{Float64}((N-2, N-2, 8));
    hess = SharedArray{Float64}((N-3, N-3, 12));

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

    for i =1:num
        X = m[i, 1:4]; #vector.
        t =  0;
        # res = X[1:2];
        @inbounds while (t < T[i]) # one optimization is fixing I,J during each step.
            t = t + dt;
            k1 = DiscreteHamilton(X             , eval, grad, p); # k = z, I,J, c, gcx, gcy, h
            k2 = DiscreteHamilton(X + k1[7]/2*dt, eval, grad, p);
            k3 = DiscreteHamilton(X + k2[7]/2*dt, eval, grad, p);
            k4 = DiscreteHamilton(X + k3[7]*dt  , eval, grad, p);

            X += (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])*dt/6.0;
            # when X is outside of extended physical domain, directly cease the ray.
            # res = [res X[1:2]]
        end
        # plot(res[1,:], res[2,:]);
        # xlabel("x");ylabel("y");
        # title("non-reflected rays");
        # axes()[:set_aspect]("equal","datalim");
        m[i,5:8] = X;
    end
    # show();
    m
end
