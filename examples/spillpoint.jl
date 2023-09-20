using SpillpointPOMDP
using POMDPDiscretization
using POMDPs
using POMDPTools
using Random
using NativeSARSOP
using POMCPOW
using ProgressMeter
using Plots
using JLD2


POMDPs.actionindex(m::SpillpointInjectionPOMDP, a) = findfirst([a] .== actions(m))

function discretize(pomdp, b; Ntraj=100, Nsamples=1, Nstates=500, Nobs=100)
    # Start by generating a good set of states
    println("Sampling initial trajectories...")
    #TODO: Maybe use these samples as the initial belief
    initial_state_samples = []
    @showprogress for i in 1:Ntraj
        s = rand(b)
        push!(initial_state_samples, s)
        stop_next = false
        while !isterminal(pomdp, s)
            if isnothing(s.x_inj)
                a = (:drill, rand(pomdp.drill_locations))
            else
                a = stop_next ? (:stop, 0) : (:inject, rand(pomdp.injection_rates))
            end
            s, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
            if sum(o[1:2]) > 0
                stop_next = true
            end
            push!(initial_state_samples, s)
        end
    end

    # Generate the histor for states
    println("Generating transitions and obserations...")
    hist = sample_transitions(pomdp, initial_state_samples, Nsamples)

    # Create state abstraction
    println("Clustering State abstraction...")
    svecs = [convert_s(Vector{Float64}, h.s, pomdp)[:] for h in hist if !isterminal(pomdp, h.s)]
    spvecs = [convert_s(Vector{Float64}, h.sp, pomdp)[:] for h in hist if !isterminal(pomdp, h.sp)]
    discrete_states = KMeansDiscretizer(Nstates)(vcat(svecs, spvecs))
    state_abstraction = NearestNeighborAbstraction(discrete_states)

    # Create a observation abstraction
    println("Clustering Observation abstraction...")
    ovecs = [convert_o(Vector{Float64}, h.s, h.a, h.o, pomdp)[:] for h in hist]
    discrete_obs = KMeansDiscretizer(Nobs)(ovecs)
    observation_abstraction = NearestNeighborAbstraction(discrete_obs)

    return DiscretizedPOMDP(pomdp, hist, state_abstraction, observation_abstraction)
end

pomdp = SpillpointInjectionPOMDP(obs_configurations=[collect(0.2:0.3:0.8), collect(0.1:0.1:0.9)], obs_rewards = [-.3, -.9])
up = SpillpointPOMDP.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointPOMDP.state2params,
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=5
)

init_states = [rand(initialstate(pomdp)) for i=1:10]

exploration_coefficient=20.
alpha_observation=0.3
k_observation=1.0
tree_queries=100

optmisitic_val_estimate(pomdp, s, h, steps) = 0.5*pomdp.trapped_reward*(trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)
solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=true, estimate_value=optmisitic_val_estimate, k_observation, alpha_observation)
planner = solve(solver, pomdp)

solver_fn(dpomdp) = solve(SARSOPSolver(), dpomdp)

resolve_histories = []
pomcpow_histories = []

for (i,s) in enumerate(init_states)
    println("case i: ", i)
    b0 = initialize_belief(up, initialstate(pomdp))
    println("Discrete solve:")
    push!(resolve_histories, simulate_with_re_solve(pomdp, up, deepcopy(b0), s, discretize, solver_fn))
    println("POMCPOW solve:")
    push!(pomcpow_histories, simulate_regular(pomdp, planner, up, deepcopy(b0), s))
end

resolve_r = [sum([h.r for h in hist]) for hist in resolve_histories]
pomcpow_r = [sum([h.r for h in hist]) for hist in pomcpow_histories]

mean(resolve_r)
mean(pompow_r)

save("resolve_histories.jld2", Dict("resolve_histories" => resolve_histories, "pomcpow_histories"=>pomcpow_histories))