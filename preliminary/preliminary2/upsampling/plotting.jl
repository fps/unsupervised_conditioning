function plot(data, ; width=680, height=200, opacity=0.5)
    img = 0.95 .* ones(Float32, height, width)
    img[1,:] .= 1f0
    img[end,:] .= 1f0
    img[:,1] .= 1f0
    img[:,end] .= 1f0
    data_max = maximum(abs.(data))
    # data_center = (data_min + data_max) / 2

    view_half_height = height / 2
    view_center = height / 2

    N_data = length(data)

    for n in 2:N_data
        d1 = data[n-1]
        d2 = data[n]

        col1 = 1+round((width-1)*(n-1.5)/N_data)
        col2 = 1+round((width-1)*(n-0.5)/N_data)

        row1 = 1+round(view_center - (view_half_height-1) * (d1 / data_max))
        row2 = 1+round(view_center - (view_half_height-1) * (d2 / data_max))

        N_steps = max(max(abs(row2-row1), (col2-col1)),1)
        #@info N_steps
        for step in 0:(1/N_steps):1
            row = Int(round(row1+(row2-row1)*step))
            col = Int(round(col1+(col2-col1)*step))
            # @info "row: $(row), col: $(col)"
            img[row, col] *= (1-opacity)
        end
    end
    Images.Gray.(img)
end

function scatter(data; opacity = 0.5, width=680, height=200, markersize=3)
    img = 0.95*ones(Float32, height, width)
    data_max = maximum(abs.(data))
    # data_center = (data_min + data_max) / 2

    view_half_height = height / 2
    view_center = height / 2

    N_data = length(data)

    for n in 1:N_data
        d = data[n]
        col_center = 1+Int(round((width-1)*(n-0.5)/N_data))
        row_center = 1+Int(round(view_center - (view_half_height-1) * (d / data_max)))

        cols = max(col_center-(markersize-1),1):min(col_center+(markersize-1), width)
        rows = max(row_center-(markersize-1),1):min(row_center+(markersize-1), height)

        img[rows, col_center] .-= opacity
        img[row_center, cols] .-= opacity
        img[row_center, col_center] += opacity
    end
    Images.Gray.(img)
end
