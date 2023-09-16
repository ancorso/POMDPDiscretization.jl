using POMDPs
using POMDPTools
using POMDPDiscretization
using Test
using POMDPModels
using Random
using NativeSARSOP
using Plots

POMDPs.actionindex(pomdp::LightDark1D, a::Int) = findfirst(a .== actions(pomdp))

pomdp = LightDark1D() 
random_policy = RandomPolicy(pomdp)

# Generate a bunch of state samples
initial_state_samples = vcat([state_hist(simulate(HistoryRecorder(), pomdp, random_policy)) for _ in 1:100]...)
hist = sample_transitions(pomdp, initial_state_samples, 10)

# Create a state abstraction
Nstates = 20
svecs = [convert_s(Vector{Float64}, h.s, pomdp) for h in hist if !isterminal(pomdp, h.s)]
spvecs = [convert_s(Vector{Float64}, h.sp, pomdp) for h in hist if !isterminal(pomdp, h.sp)]
discrete_states = KMeansDiscretizer(Nstates)([svecs..., spvecs...])
state_abstraction = NearestNeighborAbstraction(discrete_states)

# Create a observation abstraction
Nobs = 20
ovecs = [convert_o(Vector, h.o, pomdp) for h in hist]
discrete_obs = KMeansDiscretizer(Nobs)(ovecs)
observation_abstraction = NearestNeighborAbstraction(discrete_obs)

# Create the discretized pomdp
discrete_pomdp = DiscretizedPOMDP(pomdp, hist, state_abstraction, observation_abstraction)

solver = SARSOPSolver()
policy = solve(solver, discrete_pomdp)

# Simulate the policy
function gen_traj()
    s = rand(initialstate(pomdp))
    println("initial state: ", s.y)
    b = initialstate(discrete_pomdp)
    up = DiscreteUpdater(discrete_pomdp)
    hist = []
    while !isterminal(pomdp, s)
        ai = action(policy, b)
        a = actions(pomdp)[ai]
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, Random.GLOBAL_RNG)

        oab = discrete_pomdp.observation_abstraction(convert_o(Vector, o, pomdp))
        bp = update(up, b, ai, oab)
        push!(hist, (;s, a, sp, o, r))
        s = sp
        b = bp
    end
    return hist
end

# Plot the results
function plot_traj(traj)
    p = heatmap(1:length(traj), -10.0:0.1:10.0, (x,y) -> pomdp.sigma(y), xlabel="t", ylabel="y")
    plot!([h.s.y for h in traj], label="s")
    hline!([-1.0, 1.0])
    for (i, h) in enumerate(traj)
        annotate!(i, h.s.y, text(h.r, :white, :center, 10))
    end
    return p
end

plot_traj(gen_traj())
