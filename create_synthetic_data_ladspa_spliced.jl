import WAV

include("../LADSPA.jl/src/LADSPA.jl")
include("splice_signals.jl")

@info "Reading input file..."

# input, input_fs = WAV.wavread("data/test/Take1_Audio 1-1.wav")
# input, input_fs = WAV.wavread("data/nam_input_add_noise.wav")
input, input_fs = WAV.wavread("data/nam_input.wav")
# input, input_fs = WAV.wavread("signal.wav")
# input, input_fs = WAV.wavread("data/nam_input_spliced_noise.wav")

# input = input[1:(30*48000)]

test_signal = 1f-2 .* randn(4 * m_amp_offset)

@info "Splicing test signal..."
input = splice_signals(input, n -> test_signal[1:n], m_amp_offset)

# WAV.wavwrite(input, "data/nam_input_spliced_noise.wav", Fs=input_fs)

samplerate = Cint(round(input_fs))

@info "Creating plugin instances..."

d_amp = filter(x -> LADSPA.label(x) == "AmpVTS", LADSPA.descriptors())[1]
d_cabinet = filter(x -> LADSPA.label(x) == "CabinetIV", LADSPA.descriptors())[1]

amp = LADSPA.instantiate(d_amp, samplerate)
cabinet = LADSPA.instantiate(d_cabinet, samplerate)

port_index(name) = LADSPA.port_index(d_amp, name)

gain = port_index("gain")
tonestack = port_index("tonestack")
lowcut = port_index("lowcut")

bass = port_index("bass")
mid = port_index("mid")
treble = port_index("treble")

@info "Setting up buffers..."

buffers_amp = LADSPA.prepare_buffers(d_amp, samplerate, length(input))

@info "Setting up common controls..."

buffers_amp[gain] .= 0.75
buffers_amp[tonestack] .= 3
buffers_amp[lowcut] .= 0.1

#=
for n in ((1:length(supervised_controls)) .- 1)
    buffers_amp[bass][(1 + (n * supervised_chunk_size)):((n+1) * supervised_chunk_size)] .= (supervised_controls[n+1][1] ./ 2) .+ 0.5
    buffers_amp[mid][(1 + (n * supervised_chunk_size)):((n+1) * supervised_chunk_size)] .= (supervised_controls[n+1][2] ./ 2) .+ 0.5
    buffers_amp[treble][(1 + (n * supervised_chunk_size)):((n+1) * supervised_chunk_size)] .= (supervised_controls[n+1][3] ./ 2) .+ 0.5
end

UnicodePlots.lineplot(buffers_amp[bass][1:(supervised_chunk_size*length(supervised_controls))], width=:auto, title="bass") |> display
UnicodePlots.lineplot(buffers_amp[mid][1:(supervised_chunk_size*length(supervised_controls))], width=:auto, title="mid") |> display
UnicodePlots.lineplot(buffers_amp[treble][1:(supervised_chunk_size*length(supervised_controls))], width=:auto, title="treble") |> display
=#

buffers_amp[bass][1:size(supervised_controls, 2)] .= (supervised_controls[1,:] .+ 1) ./ 2
buffers_amp[mid][1:size(supervised_controls, 2)] .= (supervised_controls[2,:] .+ 1) ./ 2
buffers_amp[treble][1:size(supervised_controls, 2)] .= (supervised_controls[3,:] .+ 1) ./ 2

UnicodePlots.lineplot(buffers_amp[bass][1:size(supervised_controls, 2)], width=:auto, title="bass") |> display
UnicodePlots.lineplot(buffers_amp[mid][1:size(supervised_controls, 2)], width=:auto, title="mid") |> display
UnicodePlots.lineplot(buffers_amp[treble][1:size(supervised_controls, 2)], width=:auto, title="treble") |> display


@info "Setting up unsupervised controls..."

unsupervised_indices = (1 + size(supervised_controls, 2)):length(input)

t = (1:length(unsupervised_indices)) ./ samplerate

f1 = 1 / t[end]
p1 = 0.456

f2 = 7 / t[end]
p2 = 2.9794

f3 = 23 / t[end]
p3 = 6.3445

buffers_amp[bass][unsupervised_indices] .= 0.5 .+ 0.5 .* sin.((2 * pi * f1) .* t .+ p1) 
buffers_amp[treble][unsupervised_indices] .= 0.5 .+ 0.5 .* sin.((2 * pi * f2) .* t .+ 2) 
buffers_amp[mid][unsupervised_indices] .= 0.5 .+ 0.5 .* sin.((2 * pi * f3) .* t .+ p3) 

buffers_amp[12] .= input

buffers_cabinet = LADSPA.prepare_buffers(d_cabinet, samplerate, length(input))
buffers_cabinet[LADSPA.port_index(d_cabinet, "model")] .= 10

LADSPA.activate(amp)
LADSPA.activate(cabinet)

@info "Running plugins: amp...."

LADSPA.run(amp, buffers_amp, 64)

# WAV.wavwrite(buffers_amp[13], "data/test/Take1_Audio 1-1_AmpVTS.wav", Fs=samplerate)

buffers_cabinet[3] .= buffers_amp[13]

@info "Running plugins: cabinet...."

LADSPA.run(cabinet, buffers_cabinet, 64)

# WAV.wavwrite(buffers_cabinet[4], "data/test/Take1_Audio 1-1_AmpVTS_CabinetIV.wav", Fs=samplerate)
# WAV.wavwrite(buffers_cabinet[4], "data/synthetic/AmpVTS/nam_input_add_noise_output.wav", Fs=samplerate)
# WAV.wavwrite(buffers_cabinet[4], "signal_output.wav", Fs=samplerate)

# WAV.wavwrite(buffers_cabinet[4], "data/synthetic/AmpVTS/nam_input_spliced_noise_output.wav", Fs=samplerate)

output = buffers_cabinet[4]

@info "Closing shop..."

LADSPA.deactivate(amp)
LADSPA.deactivate(cabinet)

LADSPA.cleanup(amp)
LADSPA.cleanup(cabinet)

@info "Done."


