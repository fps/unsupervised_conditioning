@info "Initializing..."

ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "8GiB"

import Pkg

@info "Activating package..."
Pkg.activate(".")

@info "Importing packages..."
import CUDA, cuDNN, Flux, WAV, Plots, DSP, Statistics, Random, ProgressBars, UnicodePlots, FFTW, MultivariateStats, DelimitedFiles, LinearAlgebra, FastRunningMedian

# Plots.unicodeplots()
Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu
cpu = Flux.cpu

include("spectrum.jl")
include("sigreg.jl")
include("log_spectrum.jl")

@info "Setting general parameters..."
fft_size = 2048
fft_hop = div(fft_size, 4)

@info "Done."
