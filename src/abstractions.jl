abstract type Abstraction end
(a::Abstraction)(v) = error("abstraction $a not callable")
n_vals(a::Abstraction) = error("abstraction $a has not implemented n_vals")

# TODO: type things so that a non-vector call will automatically be converted to a vector call

struct NearestNeighborAbstraction <: Abstraction
    vals
    distance
    NearestNeighborAbstraction(vals; distance=(v1, v2) -> norm(v1 .- v2)) = new(vals, distance)
end

n_vals(a::NearestNeighborAbstraction) = length(a.vals)

# Get the closest discrete value to the input
(n::NearestNeighborAbstraction)(v) = argmin([n.distance(v, v2) for v2 in n.vals])



