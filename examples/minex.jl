using HDF5
using ParticleFilters
using Images
using Random
using NativeSARSOP
using POMDPDiscretization
using Plots
include("pomdps/minex.jl")

# Construct the POMDP
σ_abc = 0.05
m = MinExPOMDP(;σ_abc, drill_locations = [(i,j) for i=3:7:31 for j=3:7:31])

# Load the ore maps
s_all = imresize(h5read("examples/pomdps/ore_maps.hdf5", "X"), (32,32))
# shuffle then split
Random.seed!(0)
s_all = s_all[:,:,shuffle(1:end)]
s_test = s_all[:,:,1:100]
s_train = s_all[:,:,101:end]


function discretize(m, b; Nstates=100, Nobs=5)
    # Define the state abstraction (with discrte states)
    states = [rand(b) for i=1:Nstates]
    state_abstraction = NearestNeighborAbstraction(states, convert=(s) -> s.ore[:])

    # Define discrete observations
    os = [gen(m, rand(states), rand(m.drill_locations), Random.GLOBAL_RNG).o for i in 1:1000]
    discrete_obs = [nothing, [v[1] for v in KMeansDiscretizer(Nobs)(os)]...]
    observation_abstraction = NearestNeighborAbstraction(discrete_obs, convert = (o)-> isnothing(o) ? [-1000.] : [o])

    # Generate samples for filling in the observations
    hist = sample_transitions(m, states, 1; rng=Random.GLOBAL_RNG)

    return DiscretizedPOMDP(m, hist, state_abstraction, observation_abstraction)
end

function sarsop_solve(discrete_pomdp; solver_time=1.0)
    return POMDPs.solve(SARSOPSolver(max_time=solver_time), discrete_pomdp)
end

test_states = [MinExState(s_test[:,:,i]) for i=1:20]
Gs = []
histories = []
for (i, s0) in enumerate(test_states)
    println("i: $i, reward: $(extraction_reward(m, s0))")
    Nparticles = size(s_train, 3)
    b0 = ParticleCollection([MinExState(s_train[:,:,i]) for i in 1:Nparticles])
    up = BootstrapFilter(m, Nparticles)
    hist = simulate_with_re_solve(m, s0, up, b0, discretize, sarsop_solve, verbose=true)
    println("i: $i, reward: $(extraction_reward(m, s0))")
    push!(histories, hist)
    G = sum([h.r for h in hist])
    push!(Gs, G)
end

mean(Gs)

histogram(Gs)

as = [length(hist) for hist in histories]

bs = [[sum(h.discrete_belief.b .> 0) for h in hist] for hist in histories]

p = plot(yscale=:log)
for b in bs
    plot!(b)
end
p
histogram(as)


# resolver = ReSolver(pomdp, discretize, solver_fn, random_policy, random_policy)

hist = simulate(HistoryRecorder(max_steps=100), pomdp, resolver)

