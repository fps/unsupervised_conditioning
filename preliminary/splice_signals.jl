function splice_signals(input_signal, test_signal_generator, offset)
    N_input = length(input_signal)
    starts = 0:offset:(N_input - (3 * offset) - 1)

    input_chunks = cat([input_signal[(1:(3 * offset)) .+ start] for start in starts]..., dims=3)
    input_chunks = input_chunks[:,:,Random.randperm(size(input_chunks, 3))]

    cat([cat(input_chunks[:,:,n], test_signal_generator(3*offset), dims=1) for n in 1:size(input_chunks, 3)]..., dims=1)
end
