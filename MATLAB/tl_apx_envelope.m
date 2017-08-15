%%%
% A MATLAB implementation of algorithms from BCM17
% Written by Martin S. Copenhaver (www.mit.edu/~mcopen)
%%%


function [betar] = tl_apx_envelope(p,k,y,X,mu,lambda,tol)

    % check argument count (final argument tol is optional
    if (nargin ~= 6) && (nargin ~=7)
        disp('Incorrect number of arguments provided. Halting execution.');
        return;
    else
        if (nargin == 6)
            tol = 1e-3;
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
    gammar = zeros(p,1);
    
    prev_imp = Inf;
    prev_obj = Inf;
    cur_obj = Inf;
    
    while (prev_imp > tol)
        
        %% wrt gammar (betar fixed)
        % ***N.B.***: We do not include the special cases as detailed in
        % BCM17 in Appendix C. These are included in the julia
        % implementation, but not here because we want the Matlab
        % implementation to be as rudimentary as possible.
        
        % need to sort entries of betar
        
        
        %% wrt betar (gammar fixed)
        
        res = sortrows([abs(betar),1:p]);

        cvx_begin
            variable gammar(p)
            variable betar(p)
            variable envelop
            minimize( 0.5*betar'*X'*X*betar - y'*X*betar+dot(y,y)/2 + mu*sum(gammar)+ lambda*envelop )
            subject to
                envelop >= 0;
                gammar >= 0;
                gammar >= betar;
                gammar >= -betar;
                envelop >= sum(gammar) - k;
        cvx_end
end
