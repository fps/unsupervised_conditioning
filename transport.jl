for n in 1:size(latent_pca_new, 1);
    @info "$n"
    idx = findmin([Distances.SqEuclidean()(latent_pca_new[n,:], centers2[k]) for k in 1:length(centers2)])[2];
    @info "$idx"
    new_idx = findmax(plan[idx, :])[2];
    # new_idx = findmax(plan[:, idx])[2];
    @info "$new_idx"
    latent_pca_uni[n,:] .= centers2[new_idx];
end
