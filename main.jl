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
    #Number of months away for time until ready and for your current launch date
    states=["3", "6", "9", "12"],
    actions=["nothing", "reschedule"],
    observations=["time to ready", "cur_launch"],
    initialstate=Uniform(["3", "6", "9", "12"]),
    discount=0.95,
    transition=function (s, a)
        return (Deterministic(s))
        if a == "nothing" #stay in same state
            return Deterministic(s)
        else
            return Uniform(["3", "6", "9", "12"]) # reschedule
        end
    end,
    observation=function (s, a)
        if a == "nothing"
            return (Deterministic(s))
        end
    end,
    reward=function (s, a)
        if a == "reschedule"
            return -100
        else
            return 10
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