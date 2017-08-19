%%%
% A MATLAB implementation of exact algorithm from BCM17
% for solving trimmed Lasso problem
% Written by Martin S. Copenhaver (www.mit.edu/~mcopen)
%%%


function [betar] = tl_exact_bigM(p,k,y,X,mu,lambda,bigM,throwbinding)

    % throwbinding is an optional final argument
    if nargin == 7
        throwbinding = true;
    end

    if ( p ~= size(X,2) )
        disp('Specified p is not equal to row dimension of X. Halting execution.');
        return;
    end

    if ~( bigM >= 0 && bigM < inf )
        disp('Invalid big-M value supplied. Halting execution.');
    end

    cvx_begin
        variable gammar(p)
        variable a(p)
        variable betar(p)
        variable z(p) binary
        minimize( 0.5*betar'*X'*X*betar - y'*X*betar+dot(y,y)/2 + mu*sum(gammar)+lambda*sum(a)  )
        subject to
            gammar >= 0;
            a >= 0;
            gammar >= betar;
            gammar >= -betar;
            a >= bigM*z + gammar - bigM*ones(p,1);
            betar <= bigM;
            betar >= -bigM;
            sum(z) == p-k;
    cvx_end


    binding = false;

    for i=1:p 
        if abs(betar(i)) >= bigM - 1e-3
            binding = true;
        end
    end

    if (binding && throwbinding)
        disp('Warning: big-M constraint is binding  -- you should increase big-M and resolve. Otherwise, re-use same big-M and set optional argument `throwbinding=false`.');
        betar = NaN;
        return;
    else
        return;
    end

end
