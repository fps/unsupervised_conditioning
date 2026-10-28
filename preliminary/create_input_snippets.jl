import WAV, DelimitedFiles

@info "Creating chunks from nam_input..."
nam_input, nam_input_fs = WAV.wavread("data/nam_input.wav")

chunklength = 24000
hoplength = 2000

inputs = []
outputs = []
params = []

for start in 1:hoplength:(length(nam_input) - chunklength);
    @info "$start";
    sstart = "$start"
    s0 = "0"
    basename = "data/synthetic/AmpVTS/$(lpad(sstart, 8, s0))"
    input = nam_input[start:(start+chunklength-1),:]
    WAV.wavwrite(input, "$(basename)_in.wav", Fs=nam_input_fs)

    param = rand(3)
    open("$(basename).txt", "w") do io
        DelimitedFiles.writedlm(io, param)
    end

    run(`applyplugin $(basename)_in.wav $(basename)_out.wav caps.so AmpVTS 1 0.75 0.75 0.5 5 $(param[1]) $(param[2]) $(param[3]) 0.25 0.75 0.75 caps.so CabinetIV 12 0`)
    output, output_fs = WAV.wavread("$(basename)_out.wav")

    push!(inputs, input)
    push!(outputs, output)
    push!(params, param)

    run(`rm $(basename)_in.wav $(basename)_out.wav $(basename).txt`)
end

inputs = cat(inputs..., dims=2)
open("data/synthetic/AmpVTS/inputs.txt", "w") do io
    DelimitedFiles.writedlm(io, inputs)
end

outputs = cat(outputs..., dims=2)
open("data/synthetic/AmpVTS/outputs.txt", "w") do io
    DelimitedFiles.writedlm(io, outputs)
end

params = cat(params..., dims=2)
open("data/synthetic/AmpVTS/params.txt", "w") do io
    DelimitedFiles.writedlm(io, params)
end

