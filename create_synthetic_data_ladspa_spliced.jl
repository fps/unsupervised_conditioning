import WAV

include("../LADSPA.jl/src/LADSPA.jl")
include("splice_signals.jl")

@info "Reading input file..."

# input, input_fs = WAV.wavread("data/test/Take1_Audio 1-1.wav")
# input, input_fs = WAV.wavread("data/nam_input_add_noise.wav")
input, input_fs = WAV.wavread("data/nam_input.wav")
# input, input_fs = WAV.wavread("signal.wav")
# input, input_fs = WAV.wavread("data/nam_input_spliced_noise.wav")

test_signal = 1f-2 .* randn(4 * m_amp_offset)

@info "Splicing test signal..."
input = splice_signals(input, n -> test_signal[1:n], m_amp_offset)

WAV.wavwrite(input, "data/nam_input_spliced_noise.wav", Fs=input_fs)

samplerate = Cint(round(input_fs))

@info "Creating plugin instances..."

d_amp = filter(x -> LADSPA.label(x) == "AmpVTS", LADSPA.descriptors())[1]
d_cabinet = filter(x -> LADSPA.label(x) == "CabinetIV", LADSPA.descriptors())[1]

amp = LADSPA.instantiate(d_amp, samplerate)
cabinet = LADSPA.instantiate(d_cabinet, samplerate)

@info "Setting up buffers..."

buffers_amp = LADSPA.prepare_buffers(d_amp, samplerate, length(input))

t = (1:length(input)) ./ 48000

f1 = 1 / t[end]
p1 = 0.456

f2 = 7 / t[end]
p2 = 2.9794

f3 = 23 / t[end]
p3 = 6.3445

buffers_amp[LADSPA.port_index(d_amp, "gain")] .= 0.75
buffers_amp[LADSPA.port_index(d_amp, "tonestack")] .= 3
buffers_amp[LADSPA.port_index(d_amp, "lowcut")] .= 0.1

buffers_amp[LADSPA.port_index(d_amp, "bass")] .= 0.5 .+ 0.5 .* sin.((2 * pi * f1) .* t .+ p1) 
buffers_amp[LADSPA.port_index(d_amp, "treble")] .= 0.5 .+ 0.5 .* sin.((2 * pi * f2) .* t .+ 2) 
buffers_amp[LADSPA.port_index(d_amp, "mid")] .= 0.5 .+ 0.5 .* sin.((2 * pi * f3) .* t .+ p3) 

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
WAV.wavwrite(buffers_cabinet[4], "data/synthetic/AmpVTS/nam_input_spliced_noise_output.wav", Fs=samplerate)

@info "Closing shop..."

LADSPA.deactivate(amp)
LADSPA.deactivate(cabinet)

LADSPA.cleanup(amp)
LADSPA.cleanup(cabinet)

@info "Done."
