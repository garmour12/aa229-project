import Pkg
using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP #POMDPLinter
import Distributions: Normal
import LinearAlgebra: normalize


# Helper function: linear scaling of standard deviation
function dummySTD(x)
    return x >= 0 ? 0.125 * x : 0
end

m = QuickPOMDP(

    # States: (t_launch, t_ready) with extra substate t_launch = -1 for no set launch date
    states = [(i, j) for i in -1:24, j in 0:24],

    # Actions: update time-to-launch to [0-24] months or do nothing (-1)
    actions = [i for i in -1:24],

    # Observations: t_launch fully observable, t_ready partially observable
    observations = [(i, j) for i in -1:24, j in 0:24],

    initialstate = Deterministic((-1, 24)),  # Initial state is (-1, 24) Maybe randomize t_ready for trials?
    isterminal = function(s)                 # Terminal state is (0, 0,); product is ready and launched
        s == (0, 0)
    end,
    discount = 1,  # No discounting in SSP

    # Transition
    transition = function(s, a)
        t_launch, t_ready = s

        # Time-to-ready always decriments by one unless it is zero
        if t_ready > 0
            t_ready -= 1
        end

        if t_launch > 0
            if a == -1         # Do nothing
                t_launch -= 1
            else               # Reschedule launch date (if t_launch = 0, no rescheduling allowed)
                t_launch = a   
            end
        end

        new_s = (t_launch, t_ready)
        return Deterministic(new_s)
    end, 
    
    # Observation (unaffected by action) TODO: figure this distribution stuff out
    observation=function (a, sp)
        t_launch, t_ready = sp
        #REMEMBER: Julia is 1-indexed
        mean = t_launch
        #will scale directly with time
        std = dummySTD(mean)

        observation_dist = Normal(mean, std)
        #will be for the 25 values or the gicen time to scheduled launch; the values will sum to 1 after normalization step
        probs = []
        observations_list_copy = [(i, j) for i in -1:24, j in 0:24]
        t_ready_values = [i for i in 0:24]
        #doing this for each observation
        for obs in t_ready_values
            pdf_val = pdf(observation_dist, obs)
            #Hoping to avoid floating point errors later
            pdf_val = round(pdf_val; digits=6)
            #Evaluate the pdf of the normal exactly at the given time to ready value
            push!(probs, pdf_val)
        end

        #normalize(probs)
        #normalize list of probs so that same relation kept, but they will definitely sum to 1
        probs /= sum(probs)
        sum_probs = sum(probs)
        if abs(sum(sum_probs) - 1) > 0
            adjust = 1.0 - sum(probs)
            #@show adjust
            max_prob_idx = argmax(probs)
            probs[max_prob_idx] += adjust
        end
        println(sum(probs))
        #Hoping to account for floating point arithmetic error
        #probs_rounded = round.(probs; digits=6)
        #@show sum(probs_rounded)
        #Will be of length 250 for every possible ttsl/ttr pair
        all_observation_pair_probs = []
        for obs in observations_list_copy
            if obs[1] == t_launch
                cur_probs_idx = obs[2]
                push!(all_observation_pair_probs, probs[cur_probs_idx+1])
            else
                push!(all_observation_pair_probs, 0.0)
            end
        end
        total_prob = sum(all_observation_pair_probs)
        @show total_prob

        return SparseCat(observations_list_copy, all_observation_pair_probs)
    end, 
    
    # Reward/Cost
    reward = function (s, a)
        t_launch, t_ready = s
        timestep_cost = 0

        # No launch date set
        if t_launch == -1
            if t_ready == 0
                timestep_cost -= 500 # Ready but no launch date set = pretty large cost for missed opportunity
            end

            if a == -1
                timestep_cost -= 50  # Do nothing = small cost
            else
                timestep_cost -= 40  # Slightly smaller cost to set initial launch date
            end
        end

        # Any valid reschedule (t_launch > 0) = medium cost
        if t_launch > 0 && a != -1
            timestep_cost -= 200
        end

        if t_launch == 0 && t_ready > 0  # Missed launch date = very large cost
            timestep_cost -= 1000
        elseif t_launch < t_ready         # Small cost for seemingly being behind
            timestep_cost -= 100
        end

        """
            Metrics:
                    Total cost of the policy:
                        Higher cost means poorer decision making that lead to the final result
                    Number of times we overshoot compared to baseline
                        Greedy will probably not do this too much but will incur high cost
                    Runtime
                        Nice to have go faster if possible
        """
        return timestep_cost
    end
)

solver = SARSOPSolver()
#solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s, b, a, o, r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")