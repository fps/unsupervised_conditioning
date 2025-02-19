import VideoIO
import Flux
import CUDA
import Statistics
import Plots

# g = Flux.gpu
g = Flux.cpu
c = Flux.cpu

io = VideoIO.open("Bee1_Vid_002_Camera0.avi")
f = VideoIO.openvideo(io)
img = read(f)
img = (x -> Float32.(x.r)).(img)
# img = repeat(img, 1, 1, 1, 1)
img = img |> g
img = reshape(img, 1, length(img), 1)
img = img .- Statistics.mean(img)

positions = Float32.(repeat(cat([(row-240)/240 for col in 1:640 for row in 1:480]', [(col-320)/320 for col in 1:640 for row in 1:480]', dims=1), 1, 1, 1)) |> g

loss(m, x, y) = Flux.mse(m(x), y)

width = 64;
depth = 8;
m = Flux.Chain(
    Flux.Dense(2 => width), 
    [Flux.Dense(width=>width, Flux.leakyrelu) for n in 1:depth]..., 
    Flux.Dense(width=>1)
) |> g

opt = Flux.setup(Flux.RAdam(0.001), m);


while true; 
    losses = []; 
    for n in 1:1000; 
        print(".")
        pos = [rand(1:480) rand(1:640)]'; 
            the_loss, the_grads = Flux.withgradient(m) do model; 
                loss(model, repeat(pos ./ [480 640]', 1, 1, 1) |> g, img[pos[:,:,1:1]...]); 
            end; 
        push!(losses, the_loss); 
        Flux.update!(opt, m, the_grads[1]); 
    end; 
    println("");
    display(Statistics.mean(losses)); 
    out_img = zeros(48, 64, 1, 1)
    for row in 1:48, col in 1:64; 
        out_img[row, col, 1, 1] = 
            (m(repeat([row*10 col*10]' ./ [480 640]', 1, 1, 1) |> g) |> c)[1, 1, 1]; 
    end; 
    Plots.heatmap(out_img[:,:,1,1]) |> display; 
end
