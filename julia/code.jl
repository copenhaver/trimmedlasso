# A julia implementation of various algorithms from BCM17
# Written by Martin S. Copenhaver (www.mit.edu/~mcopen)


##########################
## Imports packages    ##
##########################

using JuMP, Gurobi, Distributions, Mosek, SCS




##################################
## Auxiliary functions          ##
##################################

function aux_lassobeta(n::Int,p::Int,k::Int,mu::Float64,lambda::Float64,XX::Array{Float64,2},loc_b_c::Array{Float64,1},grad_rest::Array{Float64,1},max_iters=10000,tol=1e-3)
  # solve subproblem wrt beta, with (outer) beta as starting point

  MAX_ITERS = max_iters;
  TOL = tol;

  lbc = copy(loc_b_c);
  lbp = loc_b_c - ones(p);
  tcur = 1./norm(XX);
  iterl = 0;

  while (iterl < MAX_ITERS) && ( norm(lbc - lbp) > TOL )
    #println(norm(lbc));
    #println(iterl," ",norm(lbc-lbp));

    lbp = lbc;

    gg = lbc - tcur*(XX*lbc + grad_rest);

    lbc = sign(gg).*max(abs(gg)-tcur*(mu+lambda)*ones(p),zeros(p));

    #tcur = TAU*tcur;

    iterl = iterl + 1;

  end
  #println(iterl);#, " ",norm(lbc));

  return(lbc);
end

function aux_admmwrtbeta(n::Int,p::Int,k::Int,mu::Float64,lambda::Float64,XX::Array{Float64,2},loc_b_c::Array{Float64,1},grad_rest::Array{Float64,1},sigma,max_iters=10000,tol=1e-3)
  # solve subproblem wrt beta, with (outer) beta as starting point

  MAX_ITERS = max_iters;
  TOL = tol;
  SIGMA = sigma;

  lbc = copy(loc_b_c);
  lbp = loc_b_c - ones(p);
  tcur = 1./norm(XX+SIGMA*eye(p));
  iterl = 0;

  while (iterl < MAX_ITERS) && ( norm(lbc - lbp) > TOL )
    #println(norm(lbc));
    #println(iterl," ",norm(lbc-lbp));

    lbp = lbc;

    gg = lbc - tcur*((XX+SIGMA*eye(p))*lbc + grad_rest);

    lbc = sign(gg).*max(abs(gg)-tcur*mu*ones(p),zeros(p));

    #tcur = TAU*tcur;

    iterl = iterl + 1;

  end
  #println(iterl);#, " ",norm(lbc));

  return(lbc);
end



##################################
## Exact methods (MIO-based)    ##
##################################

### SOS-1 formulation

function tl_exact(p,k,y,X,mu,lambda,solver)

  # Input: a covariance matrix S and the desired number of factors to perform factor analysis.
  # Other parameters: optimal algorithmic parameters with defaults
  # Output: decomposition S = T + P + N, where T is positive-semidefinite (PSD) with rank <= factors,
  #         P is diagonal and PSD, and N is PSD (N is the "noise" component)
  
  ### algorithmic parameters:
  # maxiter.* : maximum number of * iterations 
  # rho : scaling parameter in ADMM
  # tol.* : * optimality tolerance
  
  # verify that S is indeed a matrix
  
  if ( p != size(X)[2] )
    println("Specified p is not equal to row dimension of X. Halting execution.");
    return;
  end

  m = Model(solver = solver);

  @variable(m, gamma[1:p] >= 0);
  @variable(m, beta[1:p] );
  @variable(m, z[1:p], Bin);
  @variable(m, pi[1:p] >= 0);

  @constraint(m, gamma[i=1:p] .>= beta[i] );
  @constraint(m, gamma[i=1:p] .>= -beta[i] );
  @constraint(m, sum(z) == p - k );
  @constraint(m, pi .<= gamma );

  # add SOS-1 constraints to the model; if the solver supplied does not support SOS-1 constraints, JuMP will throw an error; we do not catch that here so it will raise to the user
  for i=1:p
    addSOS1(m, [z[i],pi[i]]);
  end

  # add quadratic objective; again, if the solver cannot handle such an objective, an error will be raised
  @objective(m, Min, dot(beta,.5*X'*X*beta) - dot(y,X*beta)+dot(y,y)/2
   + (mu+lambda)*sum(gamma)-lambda*sum(pi) )

  solve(m);

  return getvalue(beta);

end


### big-M formulation

function tl_exact_bigM(p,k,y,X,mu,lambda,solver,bigM,throwbinding=true)
  if ( p != size(X)[2] )
    println("Specified p is not equal to row dimension of X. Halting execution.");
    return;
  end

  if !( bigM >= 0 && bigM < Inf )
    println("Invalid big-M value supplied. Halting execution.");
  end

  m = Model(solver = solver);

  @variable(m, gamma[1:p] >= 0);
  @variable(m, a[1:p] >= 0);
  @variable(m, beta[1:p] );
  @variable(m, z[1:p], Bin);

  @constraint(m, gamma[i=1:p] .>= beta[i] );
  @constraint(m, gamma[i=1:p] .>= -beta[i] );
  @constraint(m, a[i=1:p] .>= bigM*z[i] + gamma[i] - bigM );
  @constraint(m, beta[1:p] .<= bigM );
  @constraint(m, beta[1:p] .>= -bigM );
  @constraint(m, sum(z[i] for i=1:p) == p - k );

  @objective(m, Min, dot(beta,.5*X'*X*beta) - dot(y,X*beta)+dot(y,y)/2
   + sum{mu*gamma[i]+lambda*a[i], i=1:p})

  solve(m);

  binding = false;

  for i=1:p 
    if abs(getvalue(beta[i])) >= bigM - 1e-3
      binding = true
    end
  end

  if (binding && throwbinding)
    println("\t\tWarning: big-M constraint is binding  -- you should increase big-M and resolve. Otherwise, re-use same big-M and set optional argument `throwbinding=false`.");;
  else
    return getvalue(beta);
  end
end


##################################
## Heuristic (convex) methods   ##
##################################


### alternating minimization

function tl_apx_altmin(p,k,y,X,mu,lambda,lassosolver=aux_lassobeta,max_iter=10000,rel_tol=1e-6,tau=0.9,sigma=5.,print_every=200)

  AM_ITER = max_iter;
  REL_TOL = rel_tol;
  TAU = tau;
  SIGMA = sigma;
  PRINT_EVERY = print_every; # AM will print output on every (PRINT_EVERY)th iteration

  beta = randn(p);#starter;#zeros(p);
  gamma = zeros(p);#starter;#zeros(p);

  XpX = X'*X; # can separate computation if desired

  prev_norm = 0;
  prev_obj = 0;

  for I=0:AM_ITER

    # solve wrt gamma (by sorting beta)

    II = zeros(p);
    sto = 0; # number set to "one" (really += lambda)

    bk = sort(abs(beta))[p-k+1];

    for i=1:p
      if (abs(beta[i]) > bk)
        gamma[i] = lambda*sign(beta[i]);
        sto = sto + 1;
      else
        if (abs(beta[i]) < bk)
          gamma[i] = 0;
        else
          II[i] = 1;
        end
      end
    end

    if sum(II) == 0 
      println("ERROR!");
    else
      if sum(II) == 1
        gamma[indmax(II)] = lambda*sign(beta[indmax(II)]);
        sto = sto + 1;
      else # |II| >= 2, so need to use special cases as detailed in paper's appendix
        #println(II);
        if bk > 0
          j = indmax(II); # arbitrary one from II ---> should probably choose randomly amongst them
          if dot(X[:,j],X*beta-y) + (mu+lambda)*sign(beta[j]) != 0
            gamma[j] = 0;
          else
            gamma[j] = lambda*sign(beta[j]);
            sto = sto + 1;
          end
          # assign rest of gamma
          for i=randperm(p)
            if (sto < k) && (II[i] > 0.5)
              gamma[i] = sign(randn())*lambda; 
              sto = sto + 1;
            end
          end

        else # so bk == 0
          # need to check interval containment over indices in II
          notcontained = false;
          corrindex = -1;
          corrdot = Inf;
          for i=randperm(p)
            if II[i] > 0.5 # i.e. == 1
              dp = dot(X[:,i],X*beta - y);
              if (abs(dp) > mu)
                notcontained = true;
                corrindex = i;
                corrdot = dp;
                break;
              end
            end
          end

          if notcontained
            j = corrindex;
            if corrdot > mu
              gamma[j] = -lambda;
              sto = sto + 1;
            else
              gamma[j] = lambda;
              sto = sto + 1;
            end
            # fill in rest of gamma
            for i=randperm(p)
              if (sto < k) && (II[i] > 0.5) && (i != j)
                gamma[i] = sign(randn())*lambda; 
                sto = sto + 1;
              end
            end
          else # any extreme point will do
            for i=randperm(p)
              if (sto < k) && (II[i] > 0.5)
                gamma[i] = sign(randn())*lambda; 
                sto = sto + 1;
              end
            end
          end

        end
      end
    end

    # ensure that sto == k

    if sto != k
      println("ERROR. EXTREME POINT NOT FOUND. ABORTING.");
      # println(gamma);
      # println(sto);
      # println(II);
      # println(beta);
      II(1)
    end


    # solve wrt beta

    beta = lassosolver(n,p,k,mu,lambda,XpX,beta,-X'*y- gamma);

    # perform updates as necessary

    cur_obj = .5*norm(y-X*beta)^2 + mu*norm(beta,1) +lambda*sum(sort(abs(beta))[1:p-k]);

    if abs(cur_obj-prev_obj)/(prev_obj+.01) < REL_TOL # .01 in denominator is for numerical tolerance with zero
      println(I);
      # println(cur_obj);
      # println(prev_obj);
      break; # end AM loops
    end

    prev_obj = cur_obj;

  end

  return copy(beta);


end




### ADMM


function tl_apx_admm(p,k,y,X,mu,lambda,max_iter=2000,rel_tol=1e-6,tau=0.9,sigma=1.,print_every=200)

  ADMM_ITER = max_iter;
  REL_TOL = rel_tol;
  TAU = tau;
  SIGMA = sigma;
  PRINT_EVERY = print_every; # AM will print output on every (PRINT_EVERY)th iteration


  XpX = X'*X; # can separate computation if desired


  # ADMM vars
  beta = zeros(p);#starter;#zeros(p);
  gamma = zeros(p);#starter;#zeros(p);
  q = zeros(p);

  # <solve ADMM>

  prev_norm = 0;
  prev_obj = 0;

  for I=0:ADMM_ITER

    beta = aux_admmwrtbeta(n,p,k,mu,lambda,XpX,beta,q-X'*y- SIGMA*gamma,SIGMA);;

    ### solve wrt gamma

    aux_sb = min(SIGMA/2*(beta.^2) + q.*beta+(1/2/SIGMA)*(q.^2) , (lambda^2)/(2*SIGMA)*ones(p) + lambda*abs(beta+q/SIGMA+lambda/SIGMA*ones(p)),
      (lambda^2)/(2*SIGMA)*ones(p) + lambda*abs(beta+q/SIGMA-lambda/SIGMA*ones(p)));
    sb = sort([(aux_sb[i],i) for i=1:p]);
    zz = zeros(p);
    for i=1:(p-k)
      #println(i);
      zz[sb[i][2]] = 1; 
    end

    for i=1:p
      if zz[i] == 0
        gamma[i] = copy(beta[i]) + copy(q[i])/SIGMA;
      else # zz[i] = 1
        aar = [(SIGMA/2*(beta[i]^2) + q[i]*beta[i]+(1/2/SIGMA)*(q[i]^2) ,    0 ),
             ((lambda^2)/(2*SIGMA) + lambda*abs(beta[i]+q[i]/SIGMA+lambda/SIGMA), beta[i] + q[i]/SIGMA + lambda/SIGMA),
             ((lambda^2)/(2*SIGMA) + lambda*abs(beta[i]+q[i]/SIGMA-lambda/SIGMA), beta[i] + q[i]/SIGMA - lambda/SIGMA)];
        #println(aar);
        gamma[i] = sort(aar)[1][2];
        #println(gamma[i]);
      end
    end


    q = copy(q) + SIGMA*(beta-gamma);

    cur_norm = norm(beta-gamma);
    cur_obj = .5*norm(y-X*beta)^2 + mu*norm(beta,1) +lambda*sum(sort(abs(beta))[1:p-k]);

    #println(abs(cur_norm-prev_norm)/(prev_norm+.01) ," , ", abs(cur_obj-prev_obj)/(prev_obj+.01) );
    if abs(cur_norm-prev_norm)/(prev_norm+.01) + abs(cur_obj-prev_obj)/(prev_obj+.01) < REL_TOL # .01 in denominator is for numerical tolerance with zero
      # println(I);
      break; # end ADMM loops
    end

    prev_norm = cur_norm;
    prev_obj = cur_obj;

  end

  # </ end ADMM>

  return copy(gamma);

end







### convex envelope

function tl_apx_envelope(p,k,y,X,mu,lambda,solver)

  m = Model(solver = solver);

  @defVar(m, tau >= 0);
  @defVar(m, gamma[1:p] >= 0);
  @defVar(m, beta[1:p] );
  @defVar(m, envelope >= 0);

  @addConstraint(m, gamma[i=1:p] .>= beta[i] );
  @addConstraint(m, gamma[i=1:p] .>= -beta[i] );
  @addConstraint(m, envelope >= sum{lambda*gamma[i], i=1:p} - lambda*k); #convex envelope! 
  #@addConstraint(m, norm2{y[i] - sum{X[i,j]*beta[j], j=1:p} , i=1:n} <= tau);
  @addConstraint(m, dot(beta,.5*X'*X*beta) - dot(y,X*beta)+dot(y,y)/2 <= tau);


  @setObjective(m, Min, tau + sum{mu*gamma[i], i=1:p} + envelope)

  solve(m);

  return getvalue(beta);

end



