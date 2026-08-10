import FFTW

N_samples = 100

N_window = 10000
N_shift = 5000
# N_windows = div(length(x), N_window)

window_starts = 1:N_shift:(length(x) - N_window)

spectra = []

for window_start in window_starts
    @info "window_start: $(window_start), in window_starts: $(window_starts)"
    current_spectra = []
    for sample in 1:N_samples
        print(".")
        current_spectrum = cpu(abs.(FFTW.fft(m_amp(dev(cat(randn(N_window, 1, 1), latent[window_start:(window_start+N_window-1), :, :], dims=2)))[:])))
        push!(current_spectra, current_spectrum[1:(div(length(current_spectrum), 2))])
    end
    println("")
    push!(spectra, Statistics.mean(cat(current_spectra..., dims=2), dims=2))
end

# avg_spectra = [Statistics.mean(cat(s..., dims=2), dims=2) for s in spectra]

heatmap = cat(spectra..., dims=2)
heatmap2 = log10.(heatmap); heatmap2 = (heatmap2 .- Statistics.mean(heatmap2, dims=2)) ./ Statistics.std(heatmap2, dims=2)

# Plots.heatmap(log10.(heatmap))

#=
N_bands = 5
bandwidth = div(size(heatmap, 1), N_bands)
starts = 1:bandwidth:size(heatmap, 1)

for start in starts
    UnicodePlots.lineplot(log10.(Statistics.mean(heatmap[start:min(size(heatmap, 1), (start+bandwidth)), :], dims=1))[:], width=:auto) |> display
end
=#


include("../NadarayaWatsonEstimator.jl/src/NadarayaWatsonEstimator.jl")

ts = collect((1:size(heatmap,2)) ./ 5)[:,:] |> dev;
xs = dev(1f-5 .* randn(size(heatmap, 2), 2));
opt = Flux.setup(Flux.Adam(), xs);
epoch = 1;
while true;
    @info "epoch $(epoch)";
    l, g = Flux.withgradient(xs) do xs;
        Flux.mse(dev(heatmap2'), NadarayaWatsonEstimator.predict(NadarayaWatsonEstimator.predict(ts, xs, x -> exp.(-x); leave_one_out = true), dev(heatmap2'), x -> exp(-x); leave_one_out = true)) + 0f-1 * Statistics.mean(xs.^2);
    end;
    Flux.update!(opt, xs, g[1]); @info "loss: $l";
    for n in 1:size(xs, 2);
        UnicodePlots.lineplot(cpu(NadarayaWatsonEstimator.predict(ts, xs, x -> exp.(-x))[:,n]), width=:auto) |> display;
    end; epoch += 1;
end
