import Pkg
Pkg.add("Iterators")
using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP, Iterators

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
time_to_scheduled_launch = 0:24
time_to_ready = 0:24
#Helper funcs
#Just make the belief be an exp decay func w.r.t time to scheduled launch
#Maybe do a diff scaled decay so it's not almost 0 prob by month 24; should still be a little bit
function dummyObs(x)
    return exp(-x)
end

m = QuickPOMDP(
    # States = (time to launch in months (0-24), time to ready in months (0-24)) -> build a belief over these 625 states
    # Initial state = (24,24)
    # Terminal state = (0,0) 
    #Gives combo of every possible time_to_scheduled_launch and time_to_ready as our discrete set of 25^2 states
    states=collect(Iterators.product(time_to_scheduled_launch, time_to_ready)),

    # Actions = update time to scheduled launch to (0-24) months (if 0, this means do nothing)
    actions=["DO NOTHING", 0:24],
    #Gonna have to dummy this one (pessimisitic to start with 24); can use an optimization start from prior work; punish for staying here
    initialstate=(NaN, 24),
    isterminal=(0, 0)

    # Transition T(s'|s,a):
    # Assumption: changing time to scheduled launch does not influence time to ready
    # If a = do nothing, subtract 1 from both time to scheduledlaunch and time to ready
    # If a = update time to launch, update accordingly and subtract 1 from time to ready
    discount = 0.95, transition=function (s, a)
        #Gonna need to figure what to do if one is 0 and the other isn't; what state to trans to?
        if s == (0, 0)
            return Deterministic((0, 0))
        end

        #If s[0] is nonzero and s[1] is 0, can still subtract 1
        #If s[0] is 0 and s[1] is nonzero, this is really bad and shouldn't happen... maybe throw an error?  Or automatically push?  Or just subtract anyway but don't let go neg
        if a == "Do Nothing"
            if s[0] == NaN
                return Deterministic((NaN, s[1] - 1))
            if s[0] == 0
                return Deterministic((0, s[1] - 1))
            # Guaranteed decreasing of both state params by 1 to progress in time
            next_s = (s[0] - 1, s[1] - 1)
            return Deterministic(next_s)
        else
            # a was to reset the time to scheduled launch, so write that a as 1st state param and progress time in second state param
            next_s = (a, s[1] - 1)
            return Deterministic(s[0]) # reset
            #If action do nothing
            #If in Nan and action is to do nothing, will be neg reward, but stay in same state and subtract 1 from time to ready
            #If either is 0, don't subtract
            #If either nonzero and non-NaN, subtract

            #If action not do nothing:
            #Just update launch time according to what action tells you to, subtract one from time to ready
        end

        # Observation O(o|a,s') -> updating time to launch does not influence time to ready -> O(o|s'):
        # TODO: talk to Mykel; make dummy for now
    end, observation=function (a, sp)
        prob = dummyObs(sp[0])
        #unsure how to specify the categorical dist but hopefully this works
        return SparseCat([sp[0], 1], [prob, 1 - prob])

        # Reward R(s,a): 
        # Scaled negative reward for updating time to launch
        #  Smaller negative reward for pushing time up, larger negative reward for pushing time back


        # Scaled negative reward for doing nothing when time to launch < time to ready
        # Smaller negative reward if time to launch/ready are larger; it's bad to wait until the last minute to reschedule
        # Large negative reward if time to ready = 0 and time to launch != 0
    end, reward=function (s, a)
        total_reward = n
        if s[0] == 0
        # Large positive reward for reaching terminal state (0,0), since that means we completed on time
           if s[1] == 0
            total_reward += 1000.0
        else
            total_reward -= 100.0
        end
        end
        # Small positive reward (or maybe even zero reward?) for doing nothing; better to not have to move the launch date
        if a == "Do Nothing"
            total_reward += 1.0
        else
            total_reward -= 50.0
        end
        return total_reward
    end
)

solver = SARSOPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s, b, a, o, r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")