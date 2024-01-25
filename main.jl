using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP

m = TigerPOMDP()

solver = SARSOPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")