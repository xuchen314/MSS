function [X,output]=f_multix(data_train,I_train,J_train,siz,opts)% with extrapolation with upper bound of eta 1
    addpath('Utilities')
    m=siz.m; n=siz.n;  k=siz.k; 
    len=length(I_train);
    
    eta=min(2*len/(m*n),1);
    tol=1e-5; maxit=500; show_progress=1; show_interval=20; 
    lambda=1e2;
    gate_upt=2/5;
    
    if m*n > 1e8  || len/(m*n) < 0.08 
       usemex=true;
    else
       usemex=false;    
    end
    if isfield(opts,'p_i');                   p_i       = opts.p_i;                  end
    if isfield(opts,'X');                     X         = opts.X;                    end
    if isfield(opts,'lambda');                lambda    = opts.lambda;               end
    if isfield(opts,'gate_upt');         gate_upt       = opts.gate_upt;             end
    if isfield(opts,'eta');                   eta       = opts.eta;                  end
    if isfield(opts,'usemex');                usemex    = opts.usemex;               end
    if isfield(opts,'tol');                   tol       = opts.tol;                  end
    if isfield(opts,'maxit');                 maxit     = opts.maxit;                end
    if isfield(opts,'show_progress');    show_progress  = opts.show_progress;        end
    if isfield(opts,'show_interval');    show_interval  = opts.show_interval;        end    
     
  
    

    
    num_fac=length(p_i);  % Number of factors
    data_train(data_train==0) = eps;
    if  ~usemex
        if show_progress
           fprintf('usemex is false \n'); 
        end
        W=sparse(I_train,J_train,1,m,n);
        M=sparse(I_train,J_train,data_train,m,n);
    else
        if show_progress
           fprintf('usemex is ture \n'); 
        end
        WXM=sparse(I_train,J_train,data_train,m,n); 
        data_train = data_train';
    end
    
    flagTime=tic;
    time=[];  RMSE=[]; obj_all=[];
    fval=@(sxp, WXM) lambda*sum(sxp./p_i)+ 1/2*sum(sum(WXM.*WXM));
    
    cX = eye(k); 
    for jj = 2:num_fac-1 
        cX = cX*X{jj}; 
    end
    
    if ~usemex 
         Xprod = X{1}*cX*X{end};   
         WXM = W.*(Xprod-M);
    else
         Xprod = partXY(X{1}',cX*X{end},I_train,J_train,len); 
         updateSval1(WXM, Xprod-data_train, len);
    end
    
    sxp=zeros(1,num_fac);
    for jj=1:num_fac
        sxp(jj)=p_norm(X{jj}, p_i(jj));
    end 
    obj_old=fval(sxp, WXM);
    
    X_old=X;   X_m=X;
    lipz_old=ones(1,num_fac);  
    lipz=lipz_old;
    rand_upt=rand(1,maxit);
    x_tol= zeros(1,num_fac);
    t_old=1;
     
    for i=1:maxit   
        if mod(i,6)==1                                     %Update extrapolation weight 
            t = (1+sqrt(1+4*t_old^2))/2;
            ext = (t_old-1)/t;   
            t_old=t; 
        end
        
        if rand_upt(i)<= gate_upt   % Update X{1}. Note that we shuffle the updating order, which may result in better performance  
             ext_st = min(ext,sqrt(lipz_old(1)/lipz(1)));   
             X_m{1} = X{1}+ext_st*(X{1}-X_old{1});         % Extrapolation 
             X_old{1} = X{1};  
             lipz_old(1) = lipz(1);

             cXX = cX*X{end};
             if ~usemex, 
                  Xprod = X_m{1}*cXX;   
                  WXM = W.*(Xprod-M);
             else
                  Xprod = partXY(X_m{1}',cXX,I_train,J_train,len); 
                  updateSval1(WXM, Xprod-data_train, len);
             end

             lipz_temp = Spectral_norm(cXX);
             lipz_temp = max(lipz_temp,1e-4); 
             lipz(1) = eta*lipz_temp^2;                      %Muliply eta <1 to get a locally  Lipchitz constants for better performance
             Xtemp=X_m{1}-WXM*cXX'/lipz(1);
             [X{1},sxp(1)]=Shrink_p_norm(Xtemp,lambda/p_i(1)/lipz(1), p_i(1));   
             x_tol(1) = false;
             if norm(X{1}-X_old{1},'fro') / norm(X{1},'fro') < tol
                  x_tol(1)=true;
             end
           
           
        elseif rand_upt(i)> 1-gate_upt                              % Update X{end}
            ext_end = min(ext,sqrt(lipz_old(end)/lipz(end)));
            X_m{end} = X{end}+ext_end*(X{end}-X_old{end});   
            X_old{end} = X{end};   
            lipz_old(end) = lipz(end); 
            XcX = X{1}*cX;
            if ~usemex 
                Xprod = XcX*X_m{end};   
                WXM = W.*(Xprod-M);
            else
                Xprod = partXY(XcX',X_m{end},I_train,J_train,len); 
                updateSval1(WXM, Xprod-data_train, len);
            end
            lipz_temp = Spectral_norm(XcX); 
            lipz_temp = max(lipz_temp,1e-4);
            lipz(end) = eta*lipz_temp^2;
            Xtemp = X_m{end}- XcX'* WXM/lipz(end);  
            [X{end},sxp(end)]=Shrink_p_norm(Xtemp,lambda/p_i(end)/lipz(end),p_i(end));     
            x_tol(end)=false;
            if norm(X{end}-X_old{end},'fro')/norm(X{end},'fro')< tol
                  x_tol(end)=true;
            end   
                          
        else  % Update one of the center factors.  The probalility of updating is less than that of the side facors,
              % which is because that the center factors are of small size and  contribute less to decrease the objective                                                             
            ind = ceil((rand_upt(i)-gate_upt)/(1-2*gate_upt)*(num_fac-2))+1;  % Indicate which center factor to update
            ext_c = min(ext,sqrt(lipz_old(ind)/lipz(ind)));
            X_m{ind} = X{ind}+ ext_c*(X{ind}- X_old{ind});
            X_old{ind} = X{ind}; 
            lipz_old(ind) = lipz(ind);
            
            cXl=eye(k); 
            for jj=2:ind-1,          
                cXl = cXl*X{jj};  
            end
            XcXl= X{1}*cXl;
            
            cXr=eye(k);
            for jj = ind+1: num_fac-1
                cXr= cXr*X{jj};
            end
            cXrX = cXr* X{end};
            
            if ~usemex, 
                Xprod = XcXl* X_m{ind} * cXrX;  
                WXM=W.*(Xprod-M);
            else
                Xprod=partXY(XcXl',X_m{ind}* cXrX,I_train,J_train,len); 
                updateSval1(WXM, Xprod-data_train, len);
            end
            lipz_templ = Spectral_norm(XcXl);  
            lipz_templ = max(lipz_templ,1e-4); 
            lipz_tempr = Spectral_norm(cXrX);  
            lipz_tempr = max(lipz_tempr,1e-4);  
            lipz(ind) = eta * lipz_templ^2 * lipz_tempr^2;
            Xtemp = X_m{ind}- XcXl'*WXM* cXrX'/lipz(ind);  
            [X{ind},sxp(ind)] = Shrink_p_norm(Xtemp,lambda/p_i(ind)/lipz(ind),p_i(ind));
            cX = cXl*X{ind}*cXr;

            x_tol(ind)=false;
            if norm(X{ind} - X_old{ind},'fro')/norm(X{ind},'fro')< tol
                x_tol=true;
            end
       end
       
       if (x_tol(1) && x_tol(end))  %prod(x_tol)
           break
        end
       
       
       if  i <500 || (mod(i,show_interval)==0)
           if  ~usemex, 
               Xprod=X{1}*cX*X{end};  
               WXM=W.*(Xprod-M);
           else
               Xprod=partXY(X{1}',cX*X{end},I_train,J_train,len); 
               updateSval1(WXM, Xprod-data_train, len);
           end
           obj=fval(sxp,WXM);                 
        end 

      
       if obj> obj_old && i<500               
          if rand_upt(i)<= gate_upt           %Without Extrapolation
              X{1}=X_old{1};
          elseif rand_upt(i)> 1-gate_upt
              X{end}=X_old{end};
          else
              X{ind}=X_old{ind};
          end
          rand_upt(i+1)=rand_upt(i);
          eta=min(1.1*eta,.65);
          if show_progress
                fprintf('Increasing the eta at %d th iteration, eta is %g \n', i,eta)         
          end
       else
           obj_old=obj;  
           if mod(i,10)==1                    
              obj_all=[obj_all,obj];
              time =[time, toc(flagTime)];
              if(isfield(opts, 'data_test'))
                   if ~usemex
                         RMSE =[RMSE, CompRMSEm(X{1}*cX*X{end},opts.I_test+ m*(opts.J_test-1), opts.data_test)];
                   else
                         RMSE =[RMSE, CompRMSE(X{1}*cX, X{end},opts.I_test, opts.J_test, opts.data_test)];
                   end
              end
              if toc(flagTime)>opts.maxtime
                  break
              end
            end
           if isfield(opts, 'data_test')&& show_progress  && (mod(i,show_interval)==0 || i==1)
               fprintf('In %d -th iteration,  obj is %4g and RMSE is %4g   \n',i,obj,RMSE(end)) 
           end   
       end            
    end
    
    
    output.obj_all=obj_all;
    output.time=time;
    output.rmse=RMSE;
    