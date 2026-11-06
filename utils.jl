import Printf

#=
function conv_weights_to_string(weights)
    print("Eigen::Matrix<float, $(size(weights, 3)), $(size(weights, 2))
    join(["$(Printf.@sprintf "%.25f" w)" for w in weights
end
=#

function write_model_weights(filename, m)
    open(filename, "w") do file;
        section = 0;
        print(file, "Eigen::Matrix<float, 16, 4> m0;\n")
        print(file, "m0 << $(join(["$(Printf.@sprintf "%.25f" w)" for w in m[1].weight[1,:,:][:]], ", "));\n\n")

        print(file, "Eigen::Vector<float, 16> b0;\n")
        print(file, "b0 << $(join(["$(Printf.@sprintf "%.25f" w)" for w in m[1].bias[:]], ", "));\n\n")
        
        section = 1;
        chain = m[2].layers;
        for layer in 1:10;
            print(file, "std::array<Eigen::Matrix<float, 16, 16>, 3> conv_weights$(section)_$(layer);\n\n");
            # for n in 3:-1:1;
            for n in 1:3;
                print(file, "conv_weights$(section)_$(layer)[$(n-1)] << $(join(["$(Printf.@sprintf "%.25f" w)" for w in chain[layer].weight[4-n,:,:][:]], ", "));\n\n");
            end;
            print(file, "Eigen::Vector<float, 16> b$(section)_$(layer);\nb$(section)_$(layer) << $(join(["$(Printf.@sprintf "%.25f" w)" for w in chain[layer].bias[:]], ", "));\n\n");
        end;

        section = 2;
        chain = m[3:(end-1)];
        for layer in 1:10;
            print(file, "std::array<Eigen::Matrix<float, 16, 16>, 3> conv_weights$(section)_$(layer);\n\n");
            # for n in 3:-1:1;
            for n in 1:3;
                print(file, "conv_weights$(section)_$(layer)[$(n-1)] << $(join(["$(Printf.@sprintf "%.25f" w)" for w in chain[layer].weight[4-n,:,:][:]], ", "));\n\n");
            end;
            print(file, "Eigen::Vector<float, 16> b$(section)_$(layer);\nb$(section)_$(layer) << $(join(["$(Printf.@sprintf "%.25f" w)" for w in chain[layer].bias[:]], ", "));\n\n");
        end;

        section = 3;
        print(file, "Eigen::Matrix<float, 1, 16> m3;\n")
        print(file, "m3 << $(join(["$(Printf.@sprintf "%.25f" w)" for w in m[end].weight[1,:,:][:]], ", "));\n\n")

        print(file, "Eigen::Vector<float, 1> b3;\n")
        print(file, "b3 << $(join(["$(Printf.@sprintf "%.25f" w)" for w in m[end].bias[:]], ", "));\n\n")
    end
end
