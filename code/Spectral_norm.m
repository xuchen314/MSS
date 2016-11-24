function s=Spectral_norm(Y)
  [m, n] = size(Y);
if 2*m < n
    s=sqrt(norm(Y*Y',2));
     return;
end
if m > 2*n
    s=Spectral_norm(Y');
    return
end
 s=norm(Y,2);
    


