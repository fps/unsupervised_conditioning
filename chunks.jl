function chunks(input, output, chunk_size, model_offset)
    @assert chunk_size > model_offset
    @assert size(input, 3) == 1
    @assert size(output, 3) == 1

    N = size(input, 1)
    starts = 1:(chunk_size-model_offset):(N-chunk_size)

    chunked_input = similar(input, chunk_size, size(input, 2), length(starts))
    chunked_output = similar(output, chunk_size-model_offset, size(output, 2), length(starts))

    for n in 1:length(starts)
        chunked_input[:, :, n] = input[(starts[n]+0):((starts[n]+0)+chunk_size-1), :, :]
        chunked_output[:, :, n] = output[(starts[n]+model_offset+0):(starts[n]+chunk_size+0-1), :, :]
    end

    chunked_input, chunked_output, starts
end
