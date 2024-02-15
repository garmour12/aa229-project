using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP

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

m = QuickPOMDP(
    # States = (time to launch in months (0-24), time to ready in months (0-24)) -> build a belief over these 625 states
    # Initial state = (24,24)
    # Terminal state = (0,0) 
    states=["now", "later", "much later"],

    # Actions = do nothing, update time to launch to (0-24) months
    actions=["reschedule", "don't reschedule"],
    observations=["now", "later", "much later"],
    initialstate=Uniform(["now", "later", "much later"]),

    # Transition T(s'|s,a):
    # Assumption: changing time to launch does not influence time to ready
    # If a = do nothing, subtract 1 from both time to launch and time to ready
    # If a = update time to launch, update accordingly and subtract 1 from time to ready
    discount=0.95, transition=function (s, a)
        if a == "don't reschedule"
            return Deterministic(s) # tiger stays behind the same door
        else # a door is opened
            return Uniform(["now", "later", "much later"]) # reset
        end

    # Observation O(o|a,s') -> updating time to launch does not influence time to ready -> O(o|s'):
    # TODO: talk to Mykel
    end, observation=function (a, sp)
        if a == "reschedule"
            if sp == "later"
                return SparseCat(["now", "later", "much later"], [0.0, 1.0, 0.0]) # sparse categorical distribution
            elseif sp == "much later"
                return SparseCat(["now", "later", "much later"], [0.0, 0.0, 1.0])
            else
                return SparseCat(["now", "later", "much later"], [1.0, 0.0, 0.0])
            end
        else
            return Uniform(["now", "later", "much later"])
        end

    # Reward R(s,a): 
    # Scaled negative reward for updating time to launch
        # Smaller negative reward for pushing time up, larger negative reward for pushing time back
    # Small positive reward (or maybe even zero reward?) for doing nothing; better to not have to move the launch date
    # Large positive reward for reaching terminal state (0,0)
    # Scaled negative reward for doing nothing when time to launch < time to ready
        # Smaller negative reward if time to launch/ready are larger; it's bad to wait until the last minute to reschedule
    # Large negative reward if time to ready = 0 and time to launch != 0
    end, reward=function (s, a)
        if a == "don't reschedule"
            return 10.0
        else # the tiger was escaped
            return -100.0
        end
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