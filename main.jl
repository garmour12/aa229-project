import Pkg
#Pkg.add("LinearAlgebra")
# using POMDPLinter: @show_requirements
using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP #POMDPLinter
import Distributions: Normal
import LinearAlgebra: normalize


#TIGER POMDP FOR REFERENCE
# m = QuickPOMDP(
#     states=["left", "right"],
#     actions=["left", "right", "listen"],
#     observations=["left", "right"],
#     initialstate=Uniform(["left", "right"]),
#     discount=0.95, transition=function (s, a)
#         if a == "listen"
#             return Deterministic(s) # tiger stays behind the same door
#         else # a door is opened
#             return Uniform(["left", "right"]) # reset
#         end
#     end, observation=function (a, sp)
#         if a == "listen"
#             if sp == "left"
#                 return SparseCat(["left", "right"], [0.85, 0.15]) # sparse categorical distribution
#             else
#                 return SparseCat(["right", "left"], [0.85, 0.15])
#             end
#         else
#             return Uniform(["left", "right"])
#         end
#     end, reward=function (s, a)
#         if a == "listen"
#             return -1.0
#         elseif s == a # the tiger was found
#             return -100.0
#         else # the tiger was escaped
#             return 10.0
#         end
#     end
# )
#-1 will mean that there is not yet an assigned time
# time_to_scheduled_launch = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# time_to_ready = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

#Helper funcs
#At the longest distance out (24 months), there will be the most uncertainty (1 std is 3 months quicker/slower)
#Only deal with pos x values
function dummySTD(x)
    return x >= 0 ? 0.125 * x : 0
end

m = QuickPOMDP(
    # States = (time to launch in months (0-24), time to ready in months (0-24)) -> build a belief over these 625 states
    # Initial state = (24,24)
    # Terminal state = (0,0) 
    #Gives combo of every possible time_to_scheduled_launch and time_to_ready as our discrete set of 25^2 states
    # states=collect(Iterators.product(time_to_scheduled_launch, time_to_ready)),
    # states = [(-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), (-1, 6), (-1, 7), (-1, 8), (-1, 9), (-1, 10), (-1, 11), (-1, 12), (-1, 13), (-1, 14), (-1, 15), (-1, 16), (-1, 17), (-1, 18), (-1, 19), (-1, 20), (-1, 21), (-1, 22), (-1, 23), (-1, 24), 
    # (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), 
    # (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), 
    # (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), 
    # (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), 
    # (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), 
    # (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), 
    # (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), 
    # (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), 
    # (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), 
    # (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), 
    # (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), 
    # (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (11, 24), 
    # (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24), 
    # (13, 0), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (13, 24), 
    # (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), 
    # (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), 
    # (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (16, 16), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), 
    # (17, 0), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23), (17, 24), 
    # (18, 0), (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 9), (18, 10), (18, 11), (18, 12), (18, 13), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), 
    # (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (19, 11), (19, 12), (19, 13), (19, 14), (19, 15), (19, 16), (19, 17), (19, 18), (19, 19), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24), 
    # (20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (20, 20), (20, 21), (20, 22), (20, 23), (20, 24), 
    # (21, 0), (21, 1), (21, 2), (21, 3), (21, 4), (21, 5), (21, 6), (21, 7), (21, 8), (21, 9), (21, 10), (21, 11), (21, 12), (21, 13), (21, 14), (21, 15), (21, 16), (21, 17), (21, 18), (21, 19), (21, 20), (21, 21), (21, 22), (21, 23), (21, 24), 
    # (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (22, 5), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 11), (22, 12), (22, 13), (22, 14), (22, 15), (22, 16), (22, 17), (22, 18), (22, 19), (22, 20), (22, 21), (22, 22), (22, 23), (22, 24), 
    # (23, 0), (23, 1), (23, 2), (23, 3), (23, 4), (23, 5), (23, 6), (23, 7), (23, 8), (23, 9), (23, 10), (23, 11), (23, 12), (23, 13), (23, 14), (23, 15), (23, 16), (23, 17), (23, 18), (23, 19), (23, 20), (23, 21), (23, 22), (23, 23), (23, 24), 
    # (24, 0), (24, 1), (24, 2), (24, 3), (24, 4), (24, 5), (24, 6), (24, 7), (24, 8), (24, 9), (24, 10), (24, 11), (24, 12), (24, 13), (24, 14), (24, 15), (24, 16), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21), (24, 22), (24, 23), (24, 24)]


    states=[(i, j) for i in -1:24, j in 0:24],

    # Actions = update time to scheduled launch to (0-24) months; -1 means do nothing
    actions=[i for i in -1:24],
    #fully observable state space, meaning the observations = states = belief space
    #observations=collect(Iterators.product(time_to_scheduled_launch, time_to_ready)),
    observations=[(i, j) for i in -1:24, j in 0:24],
    #Gonna have to dummy this one (pessimisitic to start with 24); can use an optimization start from prior work; punish for staying here
    initialstate=Deterministic((-1, 24)),
    isterminal=function (s)
        s == (0, 0)
    end,

    # Transition T(s'|s,a):
    # Assumption: changing time to scheduled launch does not influence time to ready
    # If a = do nothing, subtract 1 from both time to scheduledlaunch and time to ready
    # If a = update time to launch, update accordingly and subtract 1 from time to ready
    discount=1,
    transition=function (s, a)
        #defining for clarity within function
        time_to_scheduled_launch, time_to_ready = s
        """
        If time to launch = 0
            continually decrement time to ready until we reach terminal (0, 0)
        """
        if time_to_ready > 0
            time_to_ready -= 1
        end
        if time_to_scheduled_launch > 0

            #a is only -1 when nothing needs to be reassigned (basically means do nothing), so in some way or another time moves forward
            if a == -1
                #Redundantbut leaving for clarity: 
                # First if statement basically means if nothing scheduled yet and action is do nothing, it stays -1
                #Second says if it is now time to launch (ttsl == 0), at the next state it will still be time to launch so no subtraction
                # if time_to_scheduled_launch == -1
                #     time_to_scheduled_launch = -1
                # if time_to_scheduled_launch == 0
                #   time_to_scheduled_launch = 0

                #advance time when action is do nothing, but only when we have time left/when we have a deadline in mind
                time_to_scheduled_launch -= 1

                # # Guaranteed decreasing of both state params by 1 to progress in time
                # next_s = (s[0] - 1, s[1] - 1)
                # return Deterministic(next_s)
                #Any time the action is not -1/do nothing, we are reassigning ttsl
            else
                #Regardless of what state is beforehand, reassign the launch date to what the action tells you to do
                time_to_scheduled_launch = a
            end
        end
        next_s = (time_to_scheduled_launch, time_to_ready)
        return Deterministic(next_s)
        # a was to reset the time to scheduled launch, so write that a as 1st state param and progress time in second state param
        # next_s = (a, s[1] - 1)
        # return Deterministic(s[0]) # reset
        #If action do nothing
        #If in Nan and action is to do nothing, will be neg reward, but stay in same state and subtract 1 from time to ready
        #If either is 0, don't subtract
        #If either nonzero and non-NaN, subtract

        #If action not do nothing:
        #Just update launch time according to what action tells you to, subtract one from time to ready

        # Observation O(o|a,s') -> updating time to launch does not influence time to ready -> O(o|s'):
        # make dummy for now
        """
            Obs function still has error with floating point numbers in the use of an average 
        """
    end, observation=function (a, sp)
        time_to_scheduled_launch, time_to_ready = sp
        #REMEMBER: Julia is 1-indexed
        mean = time_to_scheduled_launch
        #will scale directly with time
        std = dummySTD(mean)

        observation_dist = Normal(mean, std)
        #will be for the 25 values or the gicen time to scheduled launch; the values will sum to 1 after normalization step
        probs = []
        observations_list_copy = [(i, j) for i in -1:24, j in 0:24]
        time_to_ready_values = [i for i in 0:24]
        #doing this for each observation
        for obs in time_to_ready_values
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
            if obs[1] == time_to_scheduled_launch
                cur_probs_idx = obs[2]
                push!(all_observation_pair_probs, probs[cur_probs_idx+1])
            else
                push!(all_observation_pair_probs, 0.0)
            end
        end
        total_prob = sum(all_observation_pair_probs)
        @show total_prob

        return SparseCat(observations_list_copy, all_observation_pair_probs)


        #With a uniform, the action is always 0
        #return Uniform(observations_list_copy)


        #opposite extreme would be full observability
        # probs = zeros(length(observations_list_copy))
        # idx = findfirst(x -> x == sp, observations_list_copy)
        # @show idx
        # probs[idx] = 1.0
        # return SparseCat(observations_list_copy, probs)


        # Reward R(s,a): GOING TO DO IT SSP STYLE (all in terms of positive costs (i.e. abs value of a penalty))
        #NO SCALED PUNISHMENTS YET
        # Scaled negative reward for updating time to launch
        #  Smaller negative reward for pushing time up, larger negative reward for pushing time back

        # Scaled negative reward for doing nothing when time to launch < time to ready
        # Smaller negative reward if time to launch/ready are larger; it's bad to wait until the last minute to reschedule
        # Large negative reward if time to ready = 0 and time to launch != 0
    end, reward=function (s, a)
        time_to_scheduled_launch, time_to_ready = s
        timestep_cost = 0
        #Any recschedule
        if a != -1 && time_to_scheduled_launch != -1
            timestep_cost -= 200
        end
        #WHen launch date not set yet, penalize doing nothing
        if time_to_scheduled_launch == -1
            #If not do nothing, it's ok because it's about to change;
            # medium penalty
            if a == -1
                timestep_cost -= 50
            end
        end
        if time_to_scheduled_launch == 0 && time_to_ready > 0
            #One of the worst things if it's the deadline and we now are late; need to push it back asap; otherwise that means we've reached our goal which is good so no cost
            #Large penalty
            timestep_cost -= 1000
        end
        if time_to_scheduled_launch < time_to_ready
            timestep_cost -= 100
        end

        #PSEUDO FOR REWARDS:
        #   IF RESCHEDULE, PENALIZE MEDIUM
        #   HUGE NEGATIVE FOR TIME TO LAUNCH = 0, TIME TO READY > 0
        #   OTHERWISE, SMALL PENALTY FOR TIME TO READY BEING GREATER THAN TIME TO LAUNCH
        #   COST IF NO LAUNCH DATE SET (BETWEEN SMALL MEDIUM)


        """
            Small-medium
                Time passes while time to ready less than time to launch and no reschedule
                Launch date not set yet and you do nothing
            Medium
                Any reschedule of time to launch
            Large
                Time to launch now and you're not ready/nothing change

            Metrics:
                    Total cost of the policy:
                        Higher cost means poorer decision making that lead to the final result
                    Number of times we overshoot compared to baseline
                        Greedy will probably not do this too much but will incur high cost
                    Runtime
                        Nice to have go faster if possible
        """



        # Small positive reward (or maybe even zero reward?) for doing nothing; better to not have to move the launch date
        # if a == "Do Nothing"

        # else
        #     total_reward -= 50.0
        # end
        return timestep_cost
    end
)

solver = SARSOPSolver()
#solver = QMDPSolver()
policy = solve(solver, m)

# println("Quick Launch Date POMDP")
# #@show_requirements solve(solver, m)
# try
#     @show_requirements solve(solver, m)
# catch err_msg
#     println(err_msg)
# end

rsum = 0.0
for (s, b, a, o, r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")