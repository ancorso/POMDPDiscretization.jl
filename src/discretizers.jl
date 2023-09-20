abstract type Discretizer end

struct KMeansDiscretizer <: Discretizer
    N
end

function (d::KMeansDiscretizer)(vals; rng=Random.GLOBAL_RNG)
    X = zeros(length(vals[1]), length(vals))
    for (i, v) in enumerate(vals)
        X[:,i] .= v
    end

    res = kmeans(X, d.N; rng)
    return [res.centers[:,i] for i=1:d.N]
end
