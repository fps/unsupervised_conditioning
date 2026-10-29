@info "Initializing..."

ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "8GiB"

import Pkg

@info "Activating package..."
Pkg.activate(".")

@info "Importing packages..."
import CUDA, cuDNN, Flux, WAV, Plots, DSP, Statistics, Random, ProgressBars, UnicodePlots, FFTW, MultivariateStats, DelimitedFiles, LinearAlgebra

# Plots.unicodeplots()
Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu
cpu = Flux.cpu


# include("../AmpModeling.jl/src/AmpModeling.jl")
# include("util.jl")
# include("chunks.jl")
# include("time_encodings.jl")
# include("synthetic_data.jl")
include("spectrum.jl")
include("sigreg.jl")

@info "Done."
