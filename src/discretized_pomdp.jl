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
POMDPs.initialstate(m::DiscretizedPOMDP) = uniform_belief(m)
POMDPs.transition(m::DiscretizedPOMDP, s::Int, a::Int) = m.T[s, a]
POMDPs.observation(m::DiscretizedPOMDP, a::Int, sp::Int) = m.O[a, sp]
POMDPs.reward(m::DiscretizedPOMDP, s::Int, a::Int) = m.R[s, a]

function POMDPs.gen(m::DiscretizedPOMDP, s, a, rng)
    
    sp, o, r = gen(m.pomdp, s, a, rng)

    sp = m.state_abstraction(sp)
    o = m.observation_abstraction(o)

    return (;sp, o, r)
end