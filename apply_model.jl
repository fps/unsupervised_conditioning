import Pkg
Pkg.activate(".")

import BSON
import CUDA
import Flux
import WAV

c = Flux.cpu
g = Flux.gpu
# g = c

BSON.@load "ts9-16x16.bson" model l_limits

model = model |> g
l_limits = l_limits 

x_test, fs_x_test = WAV.wavread("input.wav")

for l1 in 0:0.1:1;
  for l2 in 0:0.1:1;
    println("$(l1) $(l2)")
    ll1 = l_limits[1][1]+l1*(l_limits[1][2] - l_limits[1][1]);
    ll2 = l_limits[2][1] + l2*(l_limits[2][2] - l_limits[2][1]);
    y_test = model(reshape(cat(0.1*x_test, ll1 * ones(Float32, length(x_test), 1), ll2 * ones(Float32, length(x_test), 1), dims=2), length(x_test), 3, 1) |> g);
    f = open("output-$(l1)-$(l2).wav", "w");
    WAV.wavwrite(y_test[:,1,1]|>c, f, Fs=48000);
    close(f);                                                                 end;
end
