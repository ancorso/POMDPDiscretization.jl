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
    T = [spzeros(Nstates) for _ in 1:Nstates, _ in 1:Nactions]
    O = [spzeros(Nobs) for _ in 1:Nactions, _ in 1:Nstates]
    R = zeros(Nstates, Nactions)
    counts = zeros(Int, Nstates, Nactions)

    for (s, a, sp, o, r) in history
        s_i = isterminal(pomdp, s) ? Nstates : state_abstraction(s)
        a_i = actionindex(pomdp, a)
        sp_i = isterminal(pomdp, sp) ? Nstates : state_abstraction(sp)
        o_i = obs_abstraction(o)
        T[s_i, a_i][sp_i] += 1
        O[a_i, sp_i][o_i] += 1
        R[s_i, a_i] += r
        counts[s_i, a_i] += 1
    end
    for s in 1:Nstates, a in 1:Nactions
        T[s, a] ./= sum(T[s, a])
        O[a, s] ./= sum(O[a, s])
    end
    # T ./= sum(T, dims=3)
    # O ./= sum(O, dims=3)
    R[counts .> 0] ./= counts[counts .> 0]

    # Fill NaNs for T (this assumes that if no transition is seen then the state is terminal and small negative reward)
    for s in 1:Nstates, a in 1:Nactions
        if any(isnan.(T[s, a]))
            T[s, a] .= 0.0
            T[s, a][end] = 1.0
            R[s, a] = -0.001
        end
    end

    # Fill NaNs for O (This assumes that if no data is seen, all obs are equally likely)
    for a in 1:Nactions, s in 1:Nstates
        if any(isnan.(O[a, s]))
            O[a, s] .= 1 / Nobs
        end
    end

    # Convert distributions to sparse cat
    Tret = Array{SparseCat, 2}(undef, Nstates, Nactions)
    Oret = Array{SparseCat, 2}(undef, Nactions, Nstates)
    for s in 1:Nstates, a in 1:Nactions
        Tret[s, a] = SparseCat(1:Nstates, T[s, a])
        Oret[a, s] = SparseCat(1:Nobs, O[a, s])
    end
    return Tret, Oret, R
end


function simulate_with_re_solve(pomdp, s0, up, b0, discretize_fn, solver_fn; verbose=true, particle_threshold=10, max_steps=20)
    s = s0
    b = b0
    hist = []
    steps = 0

    discrete_pomdp = discretize_fn(pomdp, b)
    solver = solver_fn(discrete_pomdp)
    discrete_belief = initialstate(discrete_pomdp)
    discrete_updater = DiscreteUpdater(discrete_pomdp)

    while !isterminal(pomdp, s)
        steps > max_steps && break

        # If the discrete belief falls below the target number of particles, re-solve
        if sum(discrete_belief.b .> 0) < particle_threshold
            verbose && println("Number of nonzero beliefs is $(sum(discrete_belief.b .> 0)), Resolving...")
            discrete_pomdp = discretize_fn(pomdp, b)
            solver = solver_fn(discrete_pomdp)
            discrete_belief = initialstate(discrete_pomdp)
            discrete_updater = DiscreteUpdater(discrete_pomdp)
        end
        a_int = action(solver, discrete_belief)
        a = actions(pomdp)[a_int]
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        o_int = discrete_pomdp.observation_abstraction(o)
        verbose && println("a: ", a, " o: ", o, " r: ", r,)

        # Discrete belief update
        discrete_belief_p = update(discrete_updater, discrete_belief, a_int, o_int)

        # Real belief update
        bp = update(up, b, a, o)

        # Store the history
        push!(hist, (;a, o, r, b, bp, discrete_belief, discrete_belief_p))

        # Set next vals
        discrete_belief = discrete_belief_p
        b = bp
        s = sp
        steps += 1
    end
    return hist
end

function simulate_regular(pomdp, s0, discrete_pomdp, planner, up, b0; verbose=true)
    s = s0
    b = b0
    hist = []

    while !isterminal(pomdp, s)
        a_int = action(planner, b)
        a = actions(pomdp)[a_int]
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        o_int = discrete_pomdp.observation_abstraction(o)
        verbose && println("a: ", a, " o: ", o, " r: ", r,)

        bp = update(up, b, a_int, o_int)

        push!(hist, (;s, a, sp, o, r, b))
        b = bp
        s = sp
    end
    return hist
end


