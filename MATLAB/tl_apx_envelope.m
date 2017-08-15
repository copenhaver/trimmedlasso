%%%
% A MATLAB implementation of algorithms from BCM17
% Written by Martin S. Copenhaver (www.mit.edu/~mcopen)
%%%


function [betar] = tl_apx_envelope(p,k,y,X,mu,lambda)

    % throwbinding is considered an optional final argument
    if nargin ~= 6
        disp('Incorrect number of arguments provided. Halting execution.');
        return;
    end

    if ( p ~= size(X,2) )
        disp('Specified p is not equal to row dimension of X. Halting execution.');
        return;
    end

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
