struct DiscretizedPOMDP <: POMDP{Int, Any, Int}
    pomdp                   # The original POMDP

    state_abstraction       # Function for mapping states to discrete states
    observation_abstraction # Function for mapping observations to discrete observations
    Nactions                # Set of discrete actions
    
    T                       # Sparse transition matrix SxAxS -> p(s′|s,a)
    O                       # Sparse observation matrix AxSxO -> p(o|a,s)
    R                       # Sparse reward matrix SxA -> r(s,a)
end

function DiscretizedPOMDP(pomdp::P, history, s_ab::SA, obs_ab::OA) where {P<:POMDP, SA<:Abstraction, OA<:Abstraction}
    T, O, R = fill_sparse_models(pomdp, history, s_ab, obs_ab)
    DiscretizedPOMDP(pomdp, s_ab, obs_ab, length(actions(pomdp)), T, O, R)
end

POMDPs.states(m::DiscretizedPOMDP) = 1:n_vals(m.state_abstraction)+1
POMDPs.stateindex(m::DiscretizedPOMDP, s::Int) = s
POMDPs.actions(m::DiscretizedPOMDP) = 1:m.Nactions
POMDPs.actionindex(m::DiscretizedPOMDP, a::Int) = a
POMDPs.observations(m::DiscretizedPOMDP) = 1:n_vals(m.observation_abstraction)
POMDPs.obsindex(m::DiscretizedPOMDP, o::Int) = o
POMDPs.discount(m::DiscretizedPOMDP) = discount(m.pomdp)
POMDPs.discount(m::DiscretizedPOMDP, a::Int) = discount(m.pomdp)
POMDPs.isterminal(m::DiscretizedPOMDP, s::Int) = s == length(states(m))
function POMDPs.initialstate(m::DiscretizedPOMDP; Nsamples=100)
    s0_dist = initialstate(m.pomdp)
    belief_vals = zeros(n_vals(m.state_abstraction)+1)
    for i in 1:Nsamples
        s = rand(s0_dist)
        @assert !isterminal(m.pomdp, s)
        belief_vals[m.state_abstraction(convert_s(Array, s, m.pomdp)[:])] += 1
    end
    belief_vals ./= sum(belief_vals)
    return DiscreteBelief(m, belief_vals)
end

POMDPs.transition(m::DiscretizedPOMDP, s::Int, a::Int) = m.T[s, a]
POMDPs.observation(m::DiscretizedPOMDP, a::Int, sp::Int) = m.O[a, sp]
POMDPs.reward(m::DiscretizedPOMDP, s::Int, a::Int) = m.R[s, a]

function POMDPs.gen(m::DiscretizedPOMDP, s, a, rng)
    ac = actions(m.pomdp)[a]
    println("s: ", s)
    println("a: ", ac)
    sp, o, r = @gen(:sp, :o, :r)(m.pomdp, s, ac, rng)

    oab = m.observation_abstraction(convert_o(Array, s, ac, o, m.pomdp)[:])

    return (;sp, o=oab, r)
end