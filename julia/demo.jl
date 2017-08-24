# A demo of julia implementation of algorithms from BCM17
# (as contained in code.jl)
# Written by Martin S. Copenhaver (www.mit.edu/~mcopen)


##############################
## Import packages and code ##
##############################


include("code.jl");
include("example-creator.jl");


########################
## Example Parameters ##
########################

n = 100;
p = 20;
k = 10;
SNR = 10.;
seed = 1;
egclass = 1;
mu = .01;
lambda = .01;
EPS = 1e-3;

### if you have the Gurobi solver use the following:

using Gurobi
SOLVER = GurobiSolver(OutputFlag=1); # possible options of interest: OutputFlag=1,TimeLimit=100000,Heuristics=.05

#### otherwise, use the free open-source solver Couenne (uncomment following two lines):

# using CoinOptServices
# SOLVER = OsilBonminSolver();



##################################
## Create example               ##
##################################

# set seed for reproducibility

srand(1);

y, X, beta0 = instance_creator(n,p,k,SNR,egclass);


######################################
## Solve exact and heuristic models ##
######################################

beta_hat_exact = tl_exact(p,k,y,X,mu,lambda,SOLVER);

# if solver you are using cannot handle SOS-1 constraints, you may need to use the big-M formulation: tl_exact_bigM

beta_hat_altmin = tl_apx_altmin(p,k,y,X,mu,lambda);

beta_hat_admm = tl_apx_admm(p,k,y,X,mu,lambda);

beta_hat_envelope = tl_apx_envelope(p,k,y,X,mu,lambda,SOLVER);


