function similarity(X, Y)
    return cos(atan2(X[1], X[2]) - atan2(Y[1], Y[2]));
end
# align polar coordinate
function alignment(signal)
    n = length(signal);
    for i = 2:n
        if abs(signal[i] - signal[i-1]) > 1.5*π
            signal[i:n] = signal[i:n] - sign(signal[i] - signal[i-1]) * 2*π;
        end
    end
    return signal
end

function derivativeCheck(signal)
    n = length(signal);
    deriv = (signal[2 : n] - signal[1:n-1]);

    # detect the largest two jumps in derivative.

    lo = 1;hi = n;
    for i = 1:n-2
        if abs(deriv[i] - deriv[i+1]) > 0.1
            lo = i ;
            break;
        end
    end
    for i = n-1:-1:2
        if abs(deriv[i] - deriv[i-1] ) > 0.1
            hi = i+1;
            break;
        end
    end
    if abs(lo - hi) < n/8 # cheat here. if too close, aperture too small to be true.
        lo = 1;
        hi = n;
    end
    return lo, hi
end
