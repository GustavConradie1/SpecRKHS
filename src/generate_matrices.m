function [G,A,R] = generate_matrices(x,y,ker)
    N=length(x(1,:)); %should be same for y, w
    G=zeros(N,N);
    A=zeros(N,N);
    R=zeros(N,N);
    if length(x(:,1))==1
        pf = parfor_progress(N);
        pfcleanup = onCleanup(@() delete(pf));
        for i=1:N
            for j=1:N
                G(i,j)=ker(x(j),x(i));
                A(i,j)=ker(y(j),x(i));
                R(i,j)=ker(y(j),y(i));
            end
            parfor_progress(pf);
        end
    else
        pf = parfor_progress(N);
        pfcleanup = onCleanup(@() delete(pf));
        for i=1:N
            for j=1:N
                G(i,j)=ker(x(:,j),x(:,i));
                A(i,j)=ker(y(:,j),x(:,i));
                R(i,j)=ker(y(:,j),y(:,i));
            end
            parfor_progress(pf);
        end
    end
end