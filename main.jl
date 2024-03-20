import Pkg
using POMDPs, QuickPOMDPs, NativeSARSOP, POMDPModels, POMDPTools, QMDP
import POMDPTools: Deterministic
import Distributions: Normal, truncated
import LinearAlgebra: normalize


# Helper function: linear scaling of standard deviation
function LinearSTDScaling(x)
    return x >= 0 ? 0.125 * x : 0
end

m = QuickPOMDP(

    # States: (t_launch, t_ready) with extra substate t_launch = -1 for no set launch date
    states = [(i, j) for i in -1:24, j in 0:24],

    # Actions: update time-to-launch to [0-24] months or do nothing (-1)
    actions = [i for i in -1:24],

    # Observations: t_launch fully observable, t_ready partially observable
    observations = [(i, j) for i in -1:24, j in 0:24],

    initialstate = Uniform([(i, j) for i = -1, j in 12:24]), # Initial state = no set launch date, 1-2 years out from ready
    isterminal = s -> s == (0, 0),                           # Terminal state is (0, 0,); ready and launched
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

        if t_launch == -1
            if a != -1
                t_launch = a   # Set initial launch date
            end
        end

        if s == (0,0)
           return Uniform([(i, j) for i = -1, j in 12:24])  # Reset to initial state
        end
        s = (t_launch, t_ready)
        #print("State: ", s)
        #print("Action: ", a)
        return Deterministic((t_launch, t_ready))
    end, 
    
    # Observation
    observation = function(a, sp)
        t_launch = sp[1]
        t_ready = sp[2]
        mean = t_ready
        std = LinearSTDScaling(t_ready)
        obs_distribution = Normal(mean, std)
        truncated_obs_distribution = truncated(obs_distribution, 0, 24)

        print("Time to launch: ", t_launch)
        print("Time to ready: ", t_ready)
        print("Observation")
        return Uniform([(t_launch1, est_t_ready) for t_launch1 = t_launch, est_t_ready = rand(truncated_obs_distribution, 50)[1:50]])
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
        elseif t_launch < t_ready        # Small cost for seemingly being behind
            timestep_cost -= 100
        end
        
        return timestep_cost
    end
)

solver = SARSOPSolver(verbose=true)
#solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s, b, a, o, r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")