function [G,A,R] = generate_matrices_kernelized(x,y,ker)
    N=length(x(1,:));
    G=zeros(N,N);
    A=zeros(N,N);
    R=zeros(N,N);
    if length(x(:,1))==1
        pf = parfor_progress(N);
        pfcleanup = onCleanup(@() delete(pf));
        for i=1:N
            G(i,:)=ker(x,x(i));
            A(i,:)=ker(y,x(i));
            R(i,:)=ker(y,y(i));
            parfor_progress(pf);
        end
    else
        pf = parfor_progress(N);
        pfcleanup = onCleanup(@() delete(pf));
        for i=1:N
            G(i,:)=transpose(ker(x,x(:,i)));
            A(i,:)=transpose(ker(y,x(:,i)));
            R(i,:)=transpose(ker(y,y(:,i)));
            parfor_progress(pf);
        end
    end
end
