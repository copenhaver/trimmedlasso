
# A demo of julia implementation of algorithms from BCM17
# (as contained in code.jl)
# Written by Martin S. Copenhaver (www.mit.edu/~mcopen)


##########################
## Imports packages    ##
##########################

using Distributions





function instance_creator(n,p,k,SNR,egclass,seed=1)

  srand(seed)

  SS = eye(p,p);
  beta0 = zeros(p);

  if egclass == 1
    rho = 0.8;
    ir = round(p/10);
    println("WARNING: PREVIOUS LINE SHOULD BE p/k. FIX.")
    for i=1:p
      if i%ir == 1 # then beta0[i] = 1
        beta0[i] = 1;
      end
      for j = 1:p 
        SS[i,j] = rho^abs(i-j);
      end
    end

  end
  if egclass == 2
    for i=1:5
      beta0[i] = 1;
    end
  end
  if egclass == 3
    for i=1:10
      beta0[i] = 1/2 + 10/9*.95*(i-1);
    end
  end
  if egclass == 4
    for i=1:6
      beta0[i] = -14 + 4*i;
    end

  end
  if egclass == 5
    for i=1:6
      beta0[i] = 1/2 + 10/9*.95*(i-1)^5;
    end
    beta0 = beta0/norm(beta0);
  end

  ### for all, define y = Xb+eps

  #println(beta0,SS);
  sig = sqrt(beta0'*SS*beta0/SNR)[1,1];
  #println(sig);
  eps = rand(Normal(0,sig),n);
  X = rand(MvNormal(SS),n)';

  # normalize columns of X to have ell2 norm of 1

  for i=1:p
    X[:,i] = X[:,i]/norm(X[:,i]);
  end

  y = X*beta0 + eps;

  return y, X, beta0; 
end


  # ##################################
  # ## Creating validation data ##
  # ##################################

  # eps_val = rand(Normal(0,sig),n);
  # X_val = rand(MvNormal(SS),n)';

  # # normalize columns of X to have ell2 norm of 1

  # for i=1:p
  #   X_val[:,i] = X_val[:,i]/norm(X_val[:,i]);
  # end

  # y_val = X_val*beta0 + eps_val;




