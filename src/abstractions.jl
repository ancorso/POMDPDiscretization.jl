abstract type Abstraction end
(a::Abstraction)(v) = error("abstraction $a not callable")
n_vals(a::Abstraction) = error("abstraction $a has not implemented n_vals")

struct NearestNeighborAbstraction <: Abstraction
    convert
    tree
    function NearestNeighborAbstraction(vals; convert=x -> x, metric=Euclidean(), tree=KDTree)
        data = hcat([convert(v) for v in vals]...)
        new(convert, tree(data, metric))
    end
end

n_vals(a::NearestNeighborAbstraction) = length(a.tree.data)

# Get the closest discrete value to the input
function (n::NearestNeighborAbstraction)(v)
    converted_v = n.convert(v)
    knn(n.tree, converted_v, 1)[1][1]
end



