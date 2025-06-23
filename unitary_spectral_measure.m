function [meas] = unitary_spectral_measure(G,A,g,theta,order,poletype,eps)
    N=length(G);
    [V,Lambda]=eig(A,G);
    h=V'*g;
    [poles,res]=rational_kernel(order,poletype);
    P=length(theta);
    meas=zeros(P,1);
    pf = parfor_progress(P);
    pfcleanup = onCleanup(@() delete(pf));
    for k=1:P
        for j=1:order
            z=exp(1i*theta(k)-1i*eps*poles(j));
            w=(Lambda+z*eye(N))'*h;
            u=(Lambda-z*eye(N))\h;
            meas(k)=meas(k)-real(res(j)*(w'*u))/(2*pi);
        end
        parfor_progress(pf);
    end
end