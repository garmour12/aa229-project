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
    states=["now", "later", "much later"],
    actions=["reschedule", "don't reschedule"],
    observations=["now", "later", "much later"],
    initialstate=Uniform(["now", "later", "much later"]),
    discount=0.95, transition=function (s, a)
        if a == "don't reschedule"
            return Deterministic(s) # tiger stays behind the same door
        else # a door is opened
            return Uniform(["now", "later", "much later"]) # reset
        end
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
    end, reward=function (s, a)
        if a == "don't reschedule"
            return 10.0
        else # the tiger was escaped
            return -100.0
        end
    end
)

#****Won't converge for some reason; getting issues with length of the type of my 
#states (regardless of whether it's string, tuple, int, etc)

# m = QuickPOMDP(
#     # First part of state tuple is number of months until ready launch is ready right now, and second number is 
#     # currently scheduled launch date launch date
#     states=[(6, 6), (6, 12), (12, 6), (12, 12)],
#     actions=["do nothing", "reschedule"],
#     observations=[(6, 6), (6, 12), (12, 6), (12, 12)],
#     initialstate=Uniform([(6, 6), (6, 12), (12, 6), (12, 12)]),
#     discount=0.95,
#     transition=function (s, a)
#         if a == "do nothing" #stay in same state
#             return Deterministic(s)
#         else
#             return Uniform(states) # reschedule
#         end
#     end,
#     observation=function (s)
#         return (Deterministic(s))
#     end,
#     reward=function (a)
#         if a == "reschedule"
#             return -100
#         else
#             return 10
#         end
#     end
# )

solver = SARSOPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s, b, a, o, r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")