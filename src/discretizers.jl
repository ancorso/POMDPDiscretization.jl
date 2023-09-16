abstract type Discretizer end

struct KMeansDiscretizer <: Discretizer
    N
end

function (d::KMeansDiscretizer)(vals; rng=Random.GLOBAL_RNG)
    X = hcat(vals...)
    res = kmeans(X, d.N; rng)
    return [res.centers[:,i] for i=1:d.N]
end
