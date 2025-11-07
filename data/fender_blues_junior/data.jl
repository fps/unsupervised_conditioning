import WAV

function data()
    x, fs_x = WAV.wavread("data/fender_blues_junior/varying_noise_20_min_seed_0.wav")
    x = x[(48000*6):end]

    x = Float32.(x)

    y, fs_y = WAV.wavread("data/fender_blues_junior/Take6_Audio 1-1.wav")
    y = y[(48000*6):end]
    y = y ./ maximum(abs.(y))
    y = Float32.(y)
    

    FS = fs_x
    
    x, y, FS, [120, 5, 0.1], []
end
