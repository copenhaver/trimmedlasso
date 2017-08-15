
% A demo of julia implementation of algorithms from BCM17
% (as contained in code.jl)
% Written by Martin S. Copenhaver (www.mit.edu/~mcopen)

function [y, X, beta0] = instance_creator(n,p,k,SNR,egclass)
    SS = eye(p,p);
    beta0 = zeros(p,1);

    %% based on class, develop special example
    
    if egclass == 1
        rho = 0.8;
        ir = round(p/k);
        for i=1:p
            if mod(i,ir) == 1 % then beta0[i] = 1
                beta0(i) = 1;
            end
            for j = 1:p 
                SS(i,j) = rho^abs(i-j);
            end
        end
    end

    
%   if egclass == 2
%     for i=1:5
%       beta0[i] = 1;
%     end
%   end
%   if egclass == 3
%     for i=1:10
%       beta0[i] = 1/2 + 10/9*.95*(i-1);
%     end
%   end
%   if egclass == 4
%     for i=1:6
%       beta0[i] = -14 + 4*i;
%     end
% 
%   end
%   if egclass == 5
%     for i=1:6
%       beta0[i] = 1/2 + 10/9*.95*(i-1)^5;
%     end
%     beta0 = beta0/norm(beta0);
%   end

    %% for all, define y = Xb+eps

    sig = sqrt(beta0'*SS*beta0/SNR);
    eps = sig*randn(n,1);
    X = mvnrnd(zeros(n,p),SS);

    % normalize columns of X to have ell2 norm of 1

    for i=1:p
        X(:,i) = X(:,i)/norm(X(:,i));
    end

    y = X*beta0 + eps;

    return ; % y, X, beta0; 

end



