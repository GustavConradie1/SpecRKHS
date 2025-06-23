function [C] = function_to_state(x,psi)
    M=length(x(1,:));
    N=length(psi);
    PsiX=zeros(M,N);
    if length(x(:,1))==1
        for i=1:M
            for j=1:N
                PsiX(i,j)=psi{j}(x(i));
            end
        end
    else
        for i=1:M
            for j=1:N
                PsiX(i,j)=psi{j}(x(:,i));
            end
        end
    end
    C=(pinv(PsiX)*x')';
end