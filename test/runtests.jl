using POMDPs
using POMDPTools
using POMDPDiscretization
using Test
using POMDPModels
using Random

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

@test states(discrete_pomdp) == 1:Nstates+1
@test actions(discrete_pomdp) == 1:length(actions(pomdp))
@test observations(discrete_pomdp) == 1:Nobs
@test initialstate(discrete_pomdp) == uniform_belief(discrete_pomdp)

# Test that the transition matrix sums to 1
for s in states(discrete_pomdp), a in actions(discrete_pomdp)
    @test sum(transition(discrete_pomdp, s, a).probs) ≈ 1.0
end

# Test that the observation matrix sums to 1
for a in actions(discrete_pomdp), s in states(discrete_pomdp)
    @test sum(observation(discrete_pomdp, a, s).probs) ≈ 1.0
end

