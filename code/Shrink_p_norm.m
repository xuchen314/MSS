function [X,sup]=Shrink_p_norm(Y, lambda, p )
    % min_X 1/2*||Y-X||^2 + lambda*||X||_p^p
    % sup = ||X||_p^p
    [m, n] = size(Y);
    if p>2 || (p<2  && p>1)
      error('The specific "p" is not supported right now :) \n')  
    end
    if lambda< eps
       X=Y;
       sup=0;
       return
    end

    if p==2
        X=Y/(2*lambda+1);
        r=min(m,n);
        sup=norm(X,'fro')^2;
    else       
        if 2*m < n
            [U, Sigma2,V] = eig(Y*Y');
            sigma2 = diag(Sigma2);
            sigma  = sqrt(sigma2);
            sigmap = solve_Lp( sigma, lambda, p );  % Shrinkage by the Lp norm.
            sup=sum(sigmap.^(p));
            tol = n * eps(sigmap(1));
            r = sum(sigmap > tol);
            mid = sigmap(m-r+1:end) ./ sigma(m-r+1:end) ;
            X = U(:, m-r+1:end) * diag(mid) * U(:, m-r+1:end)' * Y;
            return;
        end
        if m > 2*n
            [X,sup]=Shrink_p_norm(Y', lambda, p );
             X = X';
            return;
        end
            [U,Sigma,V] = svd(Y);
            sigma = diag(Sigma);
            sigmap = solve_Lp( sigma, lambda, p );  % Shrinkage by the Lp norm.
            sup=sum(sigmap.^(p));
            tol = n * eps(sigmap(1));
            r = sum(sigmap > tol);
            X=U(:,1:r)*diag(sigmap(1:r))*V(:,1:r)';
    end
end


function   x   =  solve_Lp( y, lambda, p )
   if p==1
       J =   1;
   elseif p<1;
       J =   2;
   end
    tau   =  (2*lambda.*(1-p))^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda)^((p-1)/(2-p));
    x     =   zeros( size(y) );
    i0    =   find( y>tau );

    if length(i0)>=1
        % lambda  =   lambda(i0);
        y0    =   y(i0);
        t     =   y0;
        for  j  =  1 : J
            t    =  y0 - p*lambda.*(t).^(p-1);
        end
        x(i0)   =  max(t,0);
    end

    % f=@(x,y)lambda*x^p+1/2*(x-y)^2;
    % error=zeros(1,length(i0));
    % for j=1:length(i0)
    %   error(j)=f(0,y(i0(j)))-f(x(i0(j)),y(i0(j)));
    % end
    % error
end