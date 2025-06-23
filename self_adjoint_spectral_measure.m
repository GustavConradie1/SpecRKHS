function meas = self_adjoint_spectral_measure(A,g,x_range,order,poletype,eps)
    N=length(A);
    [poles,res]=rational_kernel(order,poletype);
    P=length(x_range);
    meas=zeros(P,1);
    pf = parfor_progress(P);
    pfcleanup = onCleanup(@() delete(pf));
    for k=1:P
        for j=1:order
            u=(A-(x_range(k)-eps*poles(j))*speye(N))\g;
            meas(k)=meas(k)-imag(res(j)*(g'*u))/(pi);
        end
        parfor_progress(pf);
    end
end