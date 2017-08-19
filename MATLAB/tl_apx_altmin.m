%%%
% A MATLAB implementation of Algorithm 1 from BCM17
% Written by Martin S. Copenhaver (www.mit.edu/~mcopen)
%%%


function [betar] = tl_apx_altmin(p,k,y,X,mu,lambda,tol,max_iters)

    % check argument count (final two arguments, tol and max_iters, are optional)
    if (nargin ~= 6) && (nargin ~=7) && (nargin ~= 8)
        disp('Incorrect number of arguments provided. Halting execution.');
        return;
    else
        if (nargin == 6)
            tol = 1e-3;
            max_iters = 1000;
        else
            if (nargin == 7)
                max_iters = 1000;
            end
        end
    end    

    if ( p ~= size(X,2) )
        disp('Specified p is not equal to row dimension of X. Halting execution.');
        return;
    end
    
    % perform alternating minimization in gamma and betar until convergence
    % criteria is satisfied
    
    % initial beta with random value
    
    betar = randn(p,1);
    
    prev_imp = Inf;
    cur_obj = Inf;
    
    iter = 0;    
    while (prev_imp > tol) && (iter < max_iters)
        iter = iter + 1;
        
        %% wrt gammar (betar fixed)
        % ***N.B.***: We do not include the special cases as detailed in
        % BCM17 in Appendix C. These are included in the julia
        % implementation, but not here because we want the Matlab
        % implementation to be as rudimentary as possible.
        
        % need to sort entries of betar
        
        res = sortrows([abs(betar)';1:p;betar']');
        
        gammar = zeros(p,1);
        for i=(p-k+1):p
            gammar(res(i,2)) = lambda*sign(res(i,3));
        end
                
        %% wrt betar (gammar fixed)
        
        cvx_begin quiet
            variable betar(p)
            minimize( 0.5*betar'*X'*X*betar - (gammar'+y'*X)*betar+dot(y,y)/2 + (mu+lambda)*sum(abs(betar)) )
        cvx_end
        
        %% update objectives
        
        prev_obj = cur_obj;
        tl_pen = sort(abs(betar));
        cur_obj = 0.5*betar'*X'*X*betar - y'*X*betar+dot(y,y)/2 + mu*sum(abs(betar)) + lambda*sum(tl_pen(1:(p-k)));
        prev_imp = prev_obj - cur_obj;
        
    end

    return;
end
