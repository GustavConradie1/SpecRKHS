function [Lambda_res,F_res,Lambda,F,res,res_verif,idx,W,W_res] = verified_eigenvalues(G,A,R,num,r)
    %if num>=1 (integer) take num eigenpairs w the lowest residuals, if
    %not take eigenpairs with residuals lower than residuals
    N=length(G);
    if nargin<5
        r=N;
    end
    [U,Sig]=eigs(G,r,'largestabs');
    Sig = sqrt(Sig); S2 = diag(1./diag(Sig));
    A2 = S2*(U'*A*U)*S2; 
    R2 = S2*(U'*R*U)*S2; R2 = (R2+R2')/2;
    [F,Lambda,W]=eig(A2);
    Lambda=diag(Lambda);
    res=zeros(r,1);
    pf = parfor_progress(r);
    pfcleanup = onCleanup(@() delete(pf));
    for i=1:r
        lambda=Lambda(i);
        g=F(:,i);
        temp=(R2-lambda*A2'-conj(lambda)*A2+norm(lambda)^2*eye(r));
        res(i)=sqrt(real(g'*temp*g/(g'*g)));
        parfor_progress(pf);
    end
    if num>=1
        [~,idx]=mink(res,num);
        Lambda_res=Lambda(idx);
        F_res=F(:,idx);
        W_res=W(:,idx);
        res_verif=res(idx);
    else
        idx=find(res<=num);
        Lambda_res=Lambda(idx);
        F_res=F(:,idx);
        W_res=W(:,idx);
        res_verif=res(idx);
    end
end