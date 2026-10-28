### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 5e62e4f2-c504-45f5-91ef-fd4138fd52f1
@debug 1

# ╔═╡ 2e571402-717e-4dec-bcb1-042a3706559b
Base.JLOptions().debug_level

# ╔═╡ 9be5d9f3-3041-4ce8-95a0-4b23fc67cc92
Logging.current_logger()

# ╔═╡ 84df7fcb-bd30-4fab-ae17-e167278b4b2f
Logging.min_enabled_level(Logging.current_logger())

# ╔═╡ 6d711a85-affd-4d60-97e1-3339facff528
device = Flux.gpu

# ╔═╡ 2e667535-7dbf-483e-a23e-919cc3019cb9
GC.gc(true); GC.gc(false); CUDA.reclaim()

# ╔═╡ 07789174-4f57-4b3a-b3fe-b75407fb9b33
mo = Flux.Conv((10,), 1 => 1) |> device;

# ╔═╡ b5eb5349-0a23-458e-9a7f-18915bbf8102
x = randn(Float32, 100, 1, 1) |> device;

# ╔═╡ 4ff8d687-2706-4131-9c95-09c5a6fe5b28
for n in 1:10000000; 
	mo(x); 
	#yield()
end;

# ╔═╡ 543c624a-d843-4d22-99c3-49fef4cdb8d2
# ╠═╡ disabled = true
#=╠═╡
cuDNN.log_messages_overflow
  ╠═╡ =#

# ╔═╡ ca63925b-fd48-4605-bb04-ed0e01b705f5
# ╠═╡ disabled = true
#=╠═╡
cuDNN.log_messages_overflow[] = false
  ╠═╡ =#

# ╔═╡ f4afa0bf-037e-4a3b-a8d6-ba3a6f8b20cf
# ╠═╡ disabled = true
#=╠═╡
cuDNN.max_log_messages
  ╠═╡ =#

# ╔═╡ 947009ba-5c93-4d4b-91c3-04fe5695c7c6
GC.gc(true)

# ╔═╡ 076ff936-dffe-45e8-9140-be95cee23f06
mo(x)

# ╔═╡ 9d37216f-5f88-4400-b984-0ecd68f3e528
# for n in 1:100000; 
# 	local y = randn(Float32, 100, 1, 1) |> device
# end;

# ╔═╡ 23c09709-3af7-4a97-8017-f2a96ec1eb7b
 varinfo(@__MODULE__(), imported=true)

# ╔═╡ 0cf362df-6690-4abc-aadf-15332d21dfaa
# ╠═╡ disabled = true
#=╠═╡
length(cuDNN.log_messages)
  ╠═╡ =#

# ╔═╡ 6dde5043-31a5-4c3a-8741-7a1ade85eb09
# ╠═╡ disabled = true
#=╠═╡
while length(cuDNN.log_messages) > 0
	popfirst!(cuDNN.log_messages)
end
  ╠═╡ =#

# ╔═╡ 341974a8-f7ce-4d9e-84d2-89b4a8c61732
# ╠═╡ disabled = true
#=╠═╡
for n in 1:1000000; popfirst!(cuDNN.log_messages); end
  ╠═╡ =#

# ╔═╡ 342ebf51-d25e-4e07-8aa9-99ab2ef5e09c
# ╠═╡ disabled = true
#=╠═╡
cuDNN.log_messages[1].str
  ╠═╡ =#

# ╔═╡ c56fa72e-5247-45dd-ba61-aadb04c61e23
# ╠═╡ disabled = true
#=╠═╡
ENV["CUDNN_LOGDEST_DBG"]
  ╠═╡ =#

# ╔═╡ 9e32b030-de36-41d3-9702-7563e9cd965e
# ╠═╡ disabled = true
#=╠═╡
ENV["CUDNN_LOGLEVEL_DBG"]
  ╠═╡ =#

# ╔═╡ 7c9a1865-dfd8-4b36-9428-0d289e7e5e45
# ╠═╡ disabled = true
#=╠═╡
cuDNN.log_message = function(a, b, c, d); end
  ╠═╡ =#

# ╔═╡ 2ba32bda-5f43-43de-a1c8-81cd5ad087ff
# ╠═╡ disabled = true
#=╠═╡
Base.JLOptions().debug_level
  ╠═╡ =#

# ╔═╡ 2e647574-ab2b-40ea-be6c-b02c57781a53
# ╠═╡ disabled = true
#=╠═╡
precompiling = ccall(:jl_generating_output, Cint, ()) != 0
  ╠═╡ =#

# ╔═╡ 01bac0a7-8be9-4099-bbf1-1a54970ee896
# ╠═╡ disabled = true
#=╠═╡
cuDNN.cudnnSetCallback(UInt(0), C_NULL, C_NULL)
  ╠═╡ =#

# ╔═╡ cc93a752-6784-462b-86b4-ef3ea1763e50
# ╠═╡ disabled = true
#=╠═╡
Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, :init, CUDA) 
  ╠═╡ =#

# ╔═╡ 91ba3bb7-0b11-4111-9af1-0af94e1c160a
begin
	import Pkg
	Pkg.activate(mktempdir())
	# Pkg.add(url="https://github.com/fps/CUDA.jl", rev="debug_fix")
	# Pkg.add(url="https://github.com/fps/CUDA.jl", subdir="lib/cudnn", rev="debug_fix")
	Pkg.add(url="https://github.com/JuliaGPU/CUDA.jl", rev="tb/log_threads")
	Pkg.add(url="https://github.com/JuliaGPU/CUDA.jl", subdir="lib/cudnn", rev="tb/log_threads")
	#Pkg.add(url="https://github.com/fps/TAC.jl#master")
	Pkg.add("Flux")
	# import TAC
	import Logging
	import Flux
	import CUDA
	import cuDNN
end
			

# ╔═╡ 94dd3efe-1242-11f0-1079-45b157b6fd92
# ╠═╡ disabled = true
#=╠═╡
import Flux, CUDA, cuDNN, Logging
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═91ba3bb7-0b11-4111-9af1-0af94e1c160a
# ╠═5e62e4f2-c504-45f5-91ef-fd4138fd52f1
# ╠═94dd3efe-1242-11f0-1079-45b157b6fd92
# ╠═2e571402-717e-4dec-bcb1-042a3706559b
# ╠═9be5d9f3-3041-4ce8-95a0-4b23fc67cc92
# ╠═84df7fcb-bd30-4fab-ae17-e167278b4b2f
# ╠═6d711a85-affd-4d60-97e1-3339facff528
# ╠═2e667535-7dbf-483e-a23e-919cc3019cb9
# ╠═07789174-4f57-4b3a-b3fe-b75407fb9b33
# ╠═b5eb5349-0a23-458e-9a7f-18915bbf8102
# ╠═4ff8d687-2706-4131-9c95-09c5a6fe5b28
# ╠═543c624a-d843-4d22-99c3-49fef4cdb8d2
# ╠═ca63925b-fd48-4605-bb04-ed0e01b705f5
# ╠═f4afa0bf-037e-4a3b-a8d6-ba3a6f8b20cf
# ╠═947009ba-5c93-4d4b-91c3-04fe5695c7c6
# ╠═076ff936-dffe-45e8-9140-be95cee23f06
# ╠═9d37216f-5f88-4400-b984-0ecd68f3e528
# ╠═23c09709-3af7-4a97-8017-f2a96ec1eb7b
# ╠═0cf362df-6690-4abc-aadf-15332d21dfaa
# ╠═6dde5043-31a5-4c3a-8741-7a1ade85eb09
# ╠═341974a8-f7ce-4d9e-84d2-89b4a8c61732
# ╠═342ebf51-d25e-4e07-8aa9-99ab2ef5e09c
# ╠═c56fa72e-5247-45dd-ba61-aadb04c61e23
# ╠═9e32b030-de36-41d3-9702-7563e9cd965e
# ╠═7c9a1865-dfd8-4b36-9428-0d289e7e5e45
# ╠═2ba32bda-5f43-43de-a1c8-81cd5ad087ff
# ╠═2e647574-ab2b-40ea-be6c-b02c57781a53
# ╠═01bac0a7-8be9-4099-bbf1-1a54970ee896
# ╠═cc93a752-6784-462b-86b4-ef3ea1763e50
