function sup=p_norm(X, p )
% ||X||_p^p
[m, n] = size(X);

if p==2
       sup = norm(X,'fro')^2;
else       
    if 2*m < n
        [U, Sigma2,V] = eig(X*X');
        sigma2 = diag(Sigma2);
        sigma  = sqrt(sigma2);
        sup = sum(sigma.^(p));
        return;
    end
    if m > 2*n
        sup = p_norm(X', p );
        return;
    end
        [U,Sigma,V] = svd(X);
        sigma = diag(Sigma);
        sup = sum(sigma.^(p));
end