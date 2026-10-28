module wright

import Flux

struct Model
  dilated_layers
  linear_layers
  linear_mixer

  function Model(depth, width, kernel_size, activation)
    dilated_layers = [ 
      Flux.SkipConnection(
        Flux.Conv((kernel_size,), (if k == 0; 1; else; width; end)=>width, activation, dilation=2^k), 
        (mx, x) -> x[(1+(kernel_size-1)*2^k):end,:,:] .+mx
      ) 
      for k in 0:(depth-1) 
    ]
    linear_layers = [ Flux.Conv((1,), width=>width) for k in 1:(depth-1) ]
    linear_mixer = Flux.Conv((1,depth), width=>1)
    new(dilated_layers, linear_layers, linear_mixer)
  end

  function Model(d, l1, l2)
    new(d, l1, l2)
  end

end

function(m::Model)(x)
  depth = length(m.dilated_layers)
  k = size(x,1)-2^depth
  outs = similar(x,k+1,size(m.linear_layers[1].weight, 2),1,0)
  for n in 1:depth
    out = m.dilated_layers[n](x)
    l = size(out,1)
    outs = cat(outs, out[(l-k):end,:,:], dims=4)
    if n < depth
      x = m.linear_layers[n](out)
    else
      x = out
    end
  end 
  m.linear_mixer(permutedims(outs, (1, 4, 2, 3)))[:,1,:,:]
end

function offset(m::Model); 2^length(m.dilated_layers) - 1; end

struct Model2
  dilated_layers
  linear_layers
  linear_mixer

  function Model2(depth, width, kernel_size, activation)
    dilated_layers = [ 
      Flux.SkipConnection(
        Flux.Conv((kernel_size,), (if k == 0; 1; else; width; end)=>width, activation, dilation=2^k), 
        (mx, x) -> x[(1+(kernel_size-1)*2^k):end,:,:] .+mx
      ) 
      for k in 0:(depth-1) 
    ]
    linear_layers = [ Flux.Conv((1,), width=>width) for k in 1:(depth-1) ]
    linear_mixer = Flux.Conv((1,), depth*width=>1)
    new(dilated_layers, linear_layers, linear_mixer)
  end

  function Model2(d, l1, l2)
    new(d, l1, l2)
  end

end

function(m::Model2)(x)
  depth = length(m.dilated_layers)
  k = size(x,1)-2^depth
  outs = similar(x,k+1,0,1)
  for n in 1:depth
    out = m.dilated_layers[n](x)
    l = size(out,1)
    outs = cat(outs, out[(l-k):end,:,:], dims=2)
    if n < depth
      x = m.linear_layers[n](out)
    else
      x = out
    end
  end 
  m.linear_mixer(outs)
end

function offset(m::Model2); 2^length(m.dilated_layers) - 1; end

end # module
