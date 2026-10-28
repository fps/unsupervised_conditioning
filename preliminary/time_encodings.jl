function fourier_basis(Ts, order)
    reshape([sin(2*pi*t*o+phase) for t in Ts, phase in [0, pi/2], o in 1:order], length(Ts), 2*order)
end

function exponential_fourier_basis(Ts, order)
    reshape([sin(2*pi*t*(2^o)+phase) for t in Ts, phase in [0, pi/2], o in 0:(order-1)], length(Ts), 2*order)
end
