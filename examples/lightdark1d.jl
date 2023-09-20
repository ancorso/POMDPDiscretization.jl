using POMDPs
using POMDPTools
using POMDPDiscretization
using Test
using POMDPModels
using Random
using NativeSARSOP
using Plots

POMDPs.actionindex(pomdp::LightDark1D, a::Int) = findfirst(a .== actions(pomdp))

function discretize(pomdp, policy, Ntraj=100, Nsamples=1, Nstates=5, Nobs=5)
    # Generate a bunch of state samples
    initial_state_samples = vcat([state_hist(simulate(HistoryRecorder(), pomdp, policy)) for _ in 1:Ntraj]...)
    hist = sample_transitions(pomdp, initial_state_samples, Nsamples)

    # Create a state abstraction
    svecs = [convert_s(Vector{Float64}, h.s, pomdp) for h in hist if !isterminal(pomdp, h.s)]
    spvecs = [convert_s(Vector{Float64}, h.sp, pomdp) for h in hist if !isterminal(pomdp, h.sp)]
    discrete_states = KMeansDiscretizer(Nstates)([svecs..., spvecs...])
    state_abstraction = NearestNeighborAbstraction(discrete_states)

    # Create a observation abstraction
    ovecs = [convert_o(Vector, h.o, pomdp) for h in hist]
    discrete_obs = KMeansDiscretizer(Nobs)(ovecs)
    observation_abstraction = NearestNeighborAbstraction(discrete_obs)

    # Create the discretized pomdp
    return DiscretizedPOMDP(pomdp, hist, state_abstraction, observation_abstraction)
end

pomdp = LightDark1D() 
random_policy = RandomPolicy(pomdp)

solver_fn(discrete_pomdp) = solve(SARSOPSolver(), discrete_pomdp)

resolver = ReSolver(pomdp, discretize, solver_fn, random_policy, random_policy)

hist = simulate(HistoryRecorder(max_steps=100), pomdp, resolver)



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
