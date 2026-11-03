@info "Setting up model..."

n_freqs = 100

encoder_activation = Flux.leakyrelu
# encoder_activation = Flux.rrelu

m_encoder = Flux.Chain(
    Flux.Dense(n_freqs => 64),
    Flux.Dense(64 => 64, encoder_activation),
    Flux.Dense(64 => 64, encoder_activation),
    Flux.Dense(64 => 64, encoder_activation),
    Flux.Dense(64 => 32, encoder_activation),
    Flux.Dense(32 => 16, encoder_activation),
    Flux.Dense(16 => 8, encoder_activation),
    Flux.Dense(8 => 3),
) |> dev

m_amp = Flux.Chain(
    Flux.Conv((1, ), 4 => 16),
    Flux.SkipConnection(Flux.Chain([Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...), (mx, x) -> (mx .+ x[(size(x, 1) - size(mx, 1) + 1):end, :, :])),
    [Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...,
    Flux.Conv((1,), 16 => 1)
) |> dev

m_amp_offset = 48000 - size(m_amp(dev(randn(Float32, 48000, 4, 1))), 1)

@info "m_amp_offset: $m_amp_offset"


@info "Setting up optimizer..."

lr = 1f-4

opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_amp])
# opt = Flux.setup(Flux.AdamW(;eta=lr, lambda=0.1), [m_encoder, m_amp])

LS = LogSpectrum.LogSpectrum(DSP.Windows.hann(m_amp_offset), 48000, 20, 10000, n_freqs)

epoch = 1

@info "Done."

# Flux.adjust!(opt[1], 2f-5)

