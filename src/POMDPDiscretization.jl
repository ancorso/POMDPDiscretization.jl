module POMDPDiscretization

using POMDPs
using POMDPTools
using Clustering
using ProgressMeter
using Random
using LinearAlgebra
using SparseArrays

export sample_transitions
include("utils.jl")

export Abstraction, NearestNeighborAbstraction, n_vals
include("abstractions.jl")

export Discretizer, KMeansDiscretizer
include("discretizers.jl")

export DiscretizedPOMDP
include("discretized_pomdp.jl")

end # module DiscretizedPOMDP
