function [res] = pseudospectra(G,A,R,grid,r)
    N=length(G);
    if nargin<5
        r=N;
    end
    [U,Sig]=eigs(G,r,'largestabs');
    Sig = sqrt(Sig); S2 = diag(1./diag(Sig));
    A2 = S2*(U'*A*U)*S2; 
    R2 = S2*(U'*R*U)*S2; R2 = (R2+R2')/2;
    l=length(grid);
    res=zeros(l,1);
    pf = parfor_progress(l);
    pfcleanup = onCleanup(@() delete(pf));
    for i=1:l
        lambda=grid(i);
        res(i)=sqrt(real(eigs(R2-lambda*A2'-conj(lambda)*A2+norm(lambda)^2*eye(r),1,'smallestabs'))); 
        parfor_progress(pf);
    end
end