function similarity(X, Y)
    return cos(atan2(X[1], X[2]) - atan2(Y[1], Y[2]));
end
# align polar coordinate
function alignment(signal)
    n = length(signal);
    for i = 2:n
        if abs(signal[i] - signal[i-1]) > π
            signal[i:n] = signal[i:n] - sign(signal[i] - signal[i-1]) * 2*π;
        end
    end
    return signal
end

function derivativeCheck(signal)
    n = length(signal);
    deriv = (signal[2 : n] - signal[1:n-1]);

    lo = 1;hi = n;
    for i = 1:n-2
        if (deriv[i] * deriv[i+1] < 0 || abs(deriv[i]) < 5e-2) && abs(deriv[i+1]) > 1.5* abs(deriv[i])
            lo = i ;
            break;
        end
    end
    for i = n-1:-1:2
        if (deriv[i] * deriv[i-1] < 0 || abs(deriv[i]) < 5e-2) && abs(deriv[i-1]) > 1.5* abs(deriv[i])
            hi = i+1;
            break;
        end
    end
    return lo, hi
end
