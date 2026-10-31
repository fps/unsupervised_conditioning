import Pkg
Pkg.activate(".")

import BSON
import CUDA
import Flux
import WAV

c = Flux.cpu
g = Flux.gpu
# g = c

# BSON.@load "ts9-16x16.bson" model l_limits

model = m_amp |> g


x_test, fs_x_test = WAV.wavread("data/test/Take1_Audio 1-1_short.wav")
x_test = Float32.(x_test)

for l1 in -1:0.4:1;
    for l2 in -1:0.4:1;
        for l3 in -1:0.4:1;
            println("$(l1) $(l2) $(l3)")
            
            y_test = model(reshape(cat(1.0*x_test, l1 * ones(Float32, length(x_test), 1), l2 * ones(Float32, length(x_test), 1), l3 * ones(Float32, length(x_test), 1), dims=2), length(x_test), 4, 1) |> g);
            f = open("data/test/output/output-take1-$(l1)-$(l2)-$(l3).wav", "w");
            WAV.wavwrite(y_test[:,1,1]|>c, f, Fs=48000);
            close(f);
            CUDA.reclaim()
        end;
    end
end
