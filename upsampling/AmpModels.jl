module AmpModels
  import Flux
  export residual_model

  struct residual_model
    layers
    function residual_model(layers)
      new(layers)
    end

    function residual_model(n_channels::Int, amp_model_width::Int)
      new([
        Flux.Conv((1,), n_channels=>amp_model_width),
            [ Flux.Conv((3,), (amp_model_width+n_channels)=>amp_model_width, Flux.tanh, dilation=d)  for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
            [Flux.Conv((3,), (amp_model_width+n_channels)=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
            Flux.Conv((1,), (amp_model_width+n_channels)=>1)
      ])
    end
  end
  
  function(m::residual_model)(data)
    input = copy(data)
    N_input = size(input, 1)
    data = m.layers[1](data)
    for n in 2:length(m.layers)
      data = m.layers[n](cat(input[(1+N_input-size(data, 1)):end,:,:], data, dims=2))
    end
    data
  end

  function model(n_channels, amp_model_widths)
    Flux.Chain(
        Flux.Conv((1,), n_channels=>amp_model_widths[1]),
        [ 
            Flux.Chain(
                Flux.Conv((3,), amp_model_widths[1]=>amp_model_widths[1], Flux.tanh, dilation=d),
                Flux.Conv((1,), amp_model_widths[1]=>amp_model_widths[1])
            )  
            for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        ]...,
        Flux.Conv((1,), amp_model_widths[1]=>amp_model_widths[2]),
        [Flux.Conv((3,), amp_model_widths[2]=>amp_model_widths[2], Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
        Flux.Conv((1,), amp_model_widths[2]=>1)
     
    )
  end
export model
end
