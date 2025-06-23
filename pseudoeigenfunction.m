function [res,V] = pseudoeigenfunction(G,A,R,lambda,r)
    N=length(G);
    if nargin<5
        r=N;
    end
    [U,Sig]=eigs(G,r,'largestabs');
    Sig = sqrt(Sig); S2 = diag(1./diag(Sig));
    A2 = S2*(U'*A*U)*S2; 
    R2 = S2*(U'*R*U)*S2; R2 = (R2+R2')/2;
    [V,eval]=eigs((R2-lambda*A2'-conj(lambda)*A2+norm(lambda)^2*eye(r)),1,'smallestabs');
    res=sqrt(real(eval));
    V = U*S2*V;
end