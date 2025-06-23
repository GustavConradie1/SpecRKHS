function [K,PsiXX,PsiXY] = edmd(x,y,w,psi)
    %each column of x should be a snapshot data vector, y=F(x) columnwise,
    %psi dictionary of observables given as cell array, w weights
    M=length(x(1,:));
    N=length(psi);
    PsiX=zeros(M,N);
    PsiY=zeros(M,N);
    if length(x(:,1))==1
        for i=1:M
            for j=1:N
                PsiX(i,j)=psi{j}(x(i));
                PsiY(i,j)=psi{j}(y(i));
            end
        end
    else
        for i=1:M
            for j=1:N
                PsiX(i,j)=psi{j}(x(:,i));
                PsiY(i,j)=psi{j}(y(:,i));
            end
        end
    end
    W=diag(w);
    PsiXX=PsiX'*W*PsiX;
    PsiXY=PsiX'*W*PsiY;
    K=(pinv(PsiXX)*PsiXY)';
end