function sample_transitions(pomdp, initial_states, N_samples_per_sa; rng=Random.GLOBAL_RNG)
    history_tuples = []
    @showprogress for s in initial_states
        for a in actions(pomdp, s)
            for _ in 1:N_samples_per_sa
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
                push!(history_tuples, (;s, a, sp, o, r))
            end
        end
    end
    return history_tuples
end

function fill_sparse_models(pomdp, history, state_abstraction, obs_abstraction)
    Nactions = length(actions(pomdp))
    Nstates = n_vals(state_abstraction)+1
    Nobs = n_vals(obs_abstraction)
    T = zeros(Nstates, Nactions, Nstates)
    O = zeros(Float64, Nactions, Nstates, Nobs)
    R = zeros(Float64, Nstates, Nactions)
    counts = zeros(Int, Nstates, Nactions)
    for (s, a, sp, o, r) in history
        s_i = isterminal(pomdp, s) ? Nstates : state_abstraction(convert_s(Vector, s, pomdp)[:])
        a_i = actionindex(pomdp, a)
        sp_i = isterminal(pomdp, sp) ? Nstates : state_abstraction(convert_s(Vector, sp, pomdp)[:])
        o_i = obs_abstraction(convert_o(Vector, s, a, o, pomdp)[:])
        T[s_i, a_i, sp_i] += 1
        O[a_i, sp_i, o_i] += 1
        R[s_i, a_i] += r
        counts[s_i, a_i] += 1
    end
    T ./= sum(T, dims=3)
    O ./= sum(O, dims=3)
    R[counts .> 0] ./= counts[counts .> 0]

    # Fill NaNs for T (this assumes that if no transition is seen then the state is terminal and a big negative reward)
    for s in 1:Nstates, a in 1:Nactions
        if any(isnan.(T[s, a, :]))
            T[s, a, :] .= 0.0
            T[s, a, end] = 1.0
            R[s, a] = -0.001
        end
    end

    # Fill NaNs for O (This assumes that if now data is seen, all obs are equally likely)
    for a in 1:Nactions, s in 1:Nstates
        if any(isnan.(O[a, s, :]))
            O[a, s, :] .= 1 / Nobs
        end
    end

    # Convert distributions to sparse cat
    Tret = Array{SparseCat, 2}(undef, Nstates, Nactions)
    Oret = Array{SparseCat, 2}(undef, Nactions, Nstates)
    for s in 1:Nstates, a in 1:Nactions
        Tret[s, a] = SparseCat(1:Nstates, T[s, a, :])
        Oret[a, s] = SparseCat(1:Nobs, O[a, s, :])
    end
    return Tret, Oret, R
end


function simulate_with_re_solve(pomdp, up, b0, s0, discretization_fn, solver_fn; verbose=true)
    b = b0
    s = s0
    hist = []

    while !isterminal(pomdp, s)
        discrete_pomdp = discretization_fn(pomdp, b)
        pol = solver_fn(discrete_pomdp)
        a = actions(pomdp)[action(pol, initialstate(discrete_pomdp))]

        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        verbose && println("a: ", a, " o: ", o, " r: ", r)

        bp = update(up, b, a, o)

        push!(hist, (;s, a, sp, o, r, b))
        b = bp
        s = sp
    end
    return hist
end

function simulate_regular(pomdp, planner, up, b0, s0; verbose=true)
    s = s0
    b = b0
    hist = []

    while !isterminal(pomdp, s)
        a = action(planner, b)
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        verbose && println("a: ", a, " o: ", o, " r: ", r)

        bp = update(up, b, a, o)

        push!(hist, (;s, a, sp, o, r, b))
        b = bp
        s = sp
    end
    return hist
end


