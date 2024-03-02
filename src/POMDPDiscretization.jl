module POMDPDiscretization

using POMDPs
using POMDPTools
using Clustering
using ProgressMeter
using Random
using LinearAlgebra
using SparseArrays
using Distances
using NearestNeighbors
using SparseArrays

export sample_transitions, simulate_with_re_solve, simulate_regular
include("utils.jl")

export Abstraction, NearestNeighborAbstraction, n_vals
include("abstractions.jl")

export Discretizer, KMeansDiscretizer
include("discretizers.jl")

export DiscretizedPOMDP
include("discretized_pomdp.jl")

end # module DiscretizedPOMDP
