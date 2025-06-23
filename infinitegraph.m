%% Computing spectral measures of Markov chain on infinite graph
addpath(genpath('./algorithms'))
clear
rng(0)

%% Parameters
N=10^4+1;
rep=10^5;
order=6;
eps=0.1;

%% generate snapshot data

x=(-(N-1)/2:1:(N-1)/2)';
y=zeros(N,rep);
p=rand(N,rep);
pf = parfor_progress(rep);
pfcleanup = onCleanup(@() delete(pf));
for j=1:rep
    for i=1:N
        if p(i,j)>2/3
            y(i,j)=x(i)-1;
        elseif p(i,j)>1/3
            y(i,j)=x(i);
        else
            y(i,j)=x(i)+1;
        end
    end
    parfor_progress(pf);
end

%% compute operator folding matrices
ker=@(x,y) kernel(x,y);
A=generate_matrices_prob(x,y,ker);

%% compute spectral measure
x_range=-0.5:0.002:1.5;
g=sparse(N,1);
g((N-1)/2)=-1/2;
g((N+3)/2)=1/2;
specmeas1=self_adjoint_spectral_measure(A,g,x_range,1,'equi',eps);
specmeas6=self_adjoint_spectral_measure(A,g,x_range,6,'equi',eps);

%% plot spectral measures
figure
x_exact=linspace(-1/3,1,100);
y_exact=(3/(4*pi))*sqrt(6*x_exact+3-9*x_exact.^2);
p1 = plot(x_range,abs(specmeas6),'linewidth',4,'Color',[0.929 0.694 0.125]);
hold on
p2 = plot(x_range,abs(specmeas1),'linewidth',3,'Color',[0.85 0.325 0.098125]);
p3 = plot(x_exact,y_exact,'linewidth',2,'Color',[0 0.447 0.741]);
box on
ylim([0 0.5])
xlim([-0.5 1.5])
title('Spectral measure of random walk','interpreter','latex','fontsize',18)
xlabel('$x$','interpreter','latex','fontsize',18)
ylabel('$[K_{\epsilon}*\mu_f](x)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
plot([-1/3,-1/3],[0,1],'--k','Linewidth',2)
plot([1,1],[0,1],'--k','Linewidth',2)
legend([p3,p2,p1],'$\rho_f(x)$','$m=1$','$m=6$','interpreter','latex','fontsize',18,'location','south')
exportgraphics(gcf,'infinite_graph_spec_meas.pdf','ContentType','vector','BackgroundColor','none')

%% generate snapshot data for compact perturbation
rng(0)

pert_size=4;
pert_prob=(rand(2*pert_size+1,2)/2-1/4)/2;
y2=zeros(N,rep);
p2=rand(N,rep);
pf = parfor_progress(rep);
pfcleanup = onCleanup(@() delete(pf));
for j=1:rep
    for i=1:N
        if (i<=(N/2+1/2)+pert_size) && (i>=(N/2+1/2)-pert_size)
            if p2(i,j)>3/4+pert_prob(i-(N/2+1/2)+pert_size+1,1)
                y2(i,j)=x(i)-1;
            elseif p2(i,j)>1/4+pert_prob(i-(N/2+1/2)+pert_size+1,2)
                y2(i,j)=x(i);
            else
                y2(i,j)=x(i)+1;
            end
        else
            if p2(i,j)>2/3
                y2(i,j)=x(i)-1;
            elseif p2(i,j)>1/3
                y2(i,j)=x(i);
            else
                y2(i,j)=x(i)+1;
            end
        end
    end
    parfor_progress(pf);
end

%% compute operator folding matrices
ker=@(x,y) kernel(x,y);
A2=generate_matrices_prob(x,y2,ker);

%% compute spectral measure
x_range=-0.5:0.0005:1.5;
specmeasb=self_adjoint_spectral_measure(A2,g,x_range,6,'equi',0.1);
specmeasc=self_adjoint_spectral_measure(A2,g,x_range,6,'equi',0.05);
specmeasd=self_adjoint_spectral_measure(A2,g,x_range,6,'equi',0.01);

%% plot spectral measures
close all
figure
plot(x_range,max(specmeasb,0),'linewidth',2,'color',[0.4940    0.1840    0.5560])
hold on
plot(x_range,max(specmeasc,0),'linewidth',2,'color',[0.4660    0.6740    0.1880])
plot(x_range,max(specmeasd,0),'linewidth',2,'color',[0.3010    0.7450    0.9330])
plot(x_range,max(specmeasc,0),'linewidth',2,'color',[0.4660    0.6740    0.1880])
plot(x_range,max(specmeasb,0),'linewidth',2,'color',[0.4940    0.1840    0.5560])
annotation('textarrow', [0.28 0.34], [0.71 0.8],'string','Eigenvalues','interpreter','latex','fontsize',12)
annotation('textarrow', [0.25 0.295], [0.66 0.47])
annotation('textarrow', [0.62 0.58], [0.6 0.4],'string','Continuous spectra','interpreter','latex','fontsize',12)
xlim([-0.5 1.5])
title('Spectral measure of compact perturbation','interpreter','latex','fontsize',18)
xlabel('$x$','interpreter','latex','fontsize',18)
ylabel('$[K_{\epsilon}*\mu_f](x)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
legend('$\epsilon=0.1$','$\epsilon=0.05$','$\epsilon=0.01$','interpreter','latex','fontsize',18,'location','northeast')
box on
exportgraphics(gcf,'infinite_graph_perturbation_spec_meas.pdf','ContentType','vector','BackgroundColor','none')

%% define kernel
function k = kernel(x,y)
    k=(x==y);
end