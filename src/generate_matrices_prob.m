function A = generate_matrices_prob(x,y,ker)
    [N,rep]=size(y);
    A=sparse(N,N);
    pf = parfor_progress(N);
    pfcleanup = onCleanup(@() delete(pf));
    for i=1:N
        A(i,i)=sum(ker(y(i,:),x(i)))/rep;
        if i~= 1
            A(i,i-1)=sum(ker(y(i-1,:),x(i)))/rep;
        end
        if i~= N
            A(i,i+1)=sum(ker(y(i+1,:),x(i)))/rep;
        end
        parfor_progress(pf);
    end
end