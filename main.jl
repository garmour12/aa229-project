import Pkg
using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP #POMDPLinter
import Distributions: Normal
import LinearAlgebra: normalize

#   HARDCODING 24 PROBABILITY LISTS TO ENSURE THEY SUM TO ONE FOR OBSERVATION FUNCTION'S SPARSECAT RETURN 

all_lists_dict = Dict(
    # Creating the first list
    0 => [0.5, 0.2, 0.15, 0.10, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the second list
    1 => [0.2, 0.5, 0.2, 0.07, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the third list
    2 => [0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the fourth list
    3 => [0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the fifth list
    4 => [0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the sixth list
    5 => [0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the seventh list
    6 => [0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the eighth list
    7 => [0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the ninth list
    8 => [0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the tenth list
    9 => [0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the eleventh list
    10 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the twelfth list
    11 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the thirteenth list
    12 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the fourteenth list
    13 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the fifteenth list
    14 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],

    # Creating the sixteenth list
    15 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0, 0],

    # Creating the seventeenth list
    16 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0, 0],

    # Creating the eighteenth list
    17 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0, 0],

    # Creating the nineteenth list
    18 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0, 0],

    # Creating the twentieth list
    19 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0, 0],

    # Creating the twenty-first list
    20 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, 0],

    # Creating the twenty-second list
    21 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0],

    # Creating the twenty-third list
    22 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05],

    # Creating the twenty-fourth list
    23 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.07, 0.2, 0.5, 0.2],

    # Creating the twenty-fifth list
    24 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.10, 0.15, 0.2, 0.5]
)

# Helper function: linear scaling of standard deviation; will be on the range of 0 to 3 given x will be between 0 and 24
function dummySTD(x)
    return x >= 0 ? 0.125 * x : 0
end

m = QuickPOMDP(

    # States: (t_launch, t_ready) with extra substate t_launch = -1 for no set launch date
    states=[(i, j) for i in -1:24, j in 0:24],

    # Actions: update time-to-launch to [0-24] months or do nothing (-1)
    actions=[i for i in -1:24],

    # Observations: t_launch fully observable, t_ready partially observable
    observations=[(i, j) for i in -1:24, j in 0:24], initialstate=Deterministic((-1, 24)),  # Initial state is (-1, 24) Maybe randomize t_ready for trials?
    obstype=Tuple,
    isterminal=function (s)                 # Terminal state is (0, 0,); product is ready and launched
        s == (0, 0)
    end,
    discount=1,  # No discounting in SSP

    # Transition
    transition=function (s, a)
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
        mean = t_ready

        probs = all_lists_dict[mean]
        #will scale directly with time
        std = dummySTD(mean)

        observation_dist = Normal(mean, std)
        # #will be for the 25 values or the gicen time to scheduled launch; the values will sum to 1 after normalization step
        # probs = []
        observations_list_copy = [(i, j) for i in -1:24, j in 0:24]
        t_ready_values = [i for i in 0:24]
        # #doing this for each observation
        # for obs in t_ready_values
        #     pdf_val = pdf(observation_dist, obs)
        #     #Hoping to avoid floating point errors later
        #     #pdf_val = round(pdf_val)
        #     #Evaluate the pdf of the normal exactly at the given time to ready value
        #     push!(probs, pdf_val)
        # end

        # #normalize(probs)
        # #normalize list of probs so that same relation kept, but they will definitely sum to 1
        # probs /= sum(probs)
        # sum_probs = sum(probs)
        # if abs(sum(sum_probs) - 1) > 0
        #     adjust = 1.0 - sum(probs)
        #     #@show adjust
        #     max_prob_idx = argmax(probs)
        #     probs[max_prob_idx] += adjust
        # end
        # println(sum(probs))
        # #Hoping to account for floating point arithmetic error
        # #probs_rounded = round.(probs; digits=6)
        # #@show sum(probs_rounded)
        # #Will be of length 250 for every possible ttsl/ttr pair
        # all_observation_pair_probs = []
        # for obs in observations_list_copy
        #     if obs[1] == t_launch
        #         cur_probs_idx = obs[2]
        #         push!(all_observation_pair_probs, probs[cur_probs_idx+1])
        #     else
        #         push!(all_observation_pair_probs, 0.0)
        #     end
        # end
        # total_prob = sum(all_observation_pair_probs)
        # @show total_prob

        # return SparseCat(observations_list_copy, all_observation_pair_probs)

        #FAILSAFE (Sike, it's buggy)
        # t_ready_probs = []
        # #doing this for each observation
        # for val in t_ready_values
        #     # lower_bound = mean - 2 * std
        #     # upper_bound = mean + 2 * std
        #     # range = upper_bound - lower_bound
        #     if val == mean
        #         push!(t_ready_probs, 0.5)
        #     end
        #     if val == mean - 1 || val == mean + 1
        #         push!(t_ready_probs, 0.2)
        #     end
        #     if val == mean - 2 || val == mean + 2
        #         push!(t_ready_probs, 0.05)

        #     else
        #         push!(t_ready_probs, 0.0)
        #     end
        # end
        # if sum(t_ready_probs) != 1
        #     error = 1 - sum(t_ready_probs)
        #     max_prob_idx = argmax(t_ready_probs)
        #     t_ready_probs[max_prob_idx] += error
        # end
        all_observation_pair_probs = zeros(Float64, 650)
        # real_probs_start = 0
        i = 1
        # start_idx = findfirst(x -> x[1] == t_launch, observations_list_copy)
        # all_observation_pair_probs[start_idx:start_idx+length(probs)-1] = probs
        for obs in observations_list_copy
            if obs[1] == t_launch
                start_idx = i
                @show start_idx
                @show obs[1]
                all_observation_pair_probs[start_idx:start_idx+length(probs)-1] = probs
            end
            i += 1
        end
        #         cur_probs_idx = obs[2]
        #         push!(all_observation_pair_probs, probs[cur_probs_idx+1])
        #     else
        #         push!(all_observation_pair_probs, 0.0)
        #     end
        # end
        # @show sum(all_observation_pair_probs)



        return SparseCat(observations_list_copy, all_observation_pair_probs)
        # if val >= lower_bound || val <= upper_bound
        #     push!(t_ready_probs, 0.5 * (1 - abs(mean - val)))

        #pdf_val = pdf(observation_dist, obs)
        #Hoping to avoid floating point errors later
        #pdf_val = round(pdf_val)
        #Evaluate the pdf of the normal exactly at the given time to ready value
        #push!(probs, pdf_val)

    end,

    # Reward/Cost
    reward=function (s, a)
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