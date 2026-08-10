import StatsBase

found_lags = [];

for n in 1:100000;
    p = n*41000;
    n = 2000; range = (1:n) .+ p;
    lags = div.(-n:n, 2);
    cr = StatsBase.crosscor(x[range], y[range], lags);
    #= UnicodePlots.lineplot(lags, cr, width=:auto) |> display; =#
    max_cr = findmax(abs.(cr));
    lags[max_cr[2]] |> display;
    push!(found_lags, lags[max_cr[2]]);
end;

minimum(found_lags) |> display
