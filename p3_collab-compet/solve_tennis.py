from GES import general_environment_solver

ges = general_environment_solver()

# with the general solver, get the agents ready
ges.ready_agents()

# allow the solver to solve the environment
ges.run_ddpg()
