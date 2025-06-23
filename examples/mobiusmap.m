%% Computing spectral measure for a rotation and a general Mobius map
addpath(genpath('./algorithms'))
rng(3)

%% generate snapshot data
N=400;
r=(rand(N,1)).^(1/4);
theta=2*pi*rand(N,1);
x=r.*exp(1i*theta);
y=zeros(N,1);
y2=zeros(N,1);
for i=1:N
    y(i)=mobius(x(i));
    y2(i)=mobius2(x(i));
end
x=x.'; y=y.'; y2=y2.';

%% compute operator folding matrices for T_1
ker=@(x,t) kernel(x,t,5);
[G,A,~]=generate_matrices(x,y,ker);
G=(G+G')/2; %force self-adjointness

%% compute spectral measures
theta=linspace(-pi,pi,200);
g=rand(N,1);
specmeas=unitary_spectral_measure(G,A,g,theta,6,'equi',0.01);

%% plot spectral measures
specmeas=specmeas/(pi*mean(specmeas));
figure
plot(theta,specmeas,'linewidth',2)
ylim([0 2])
xlim([-pi pi])
xticks([-pi -2*pi/3 -pi/3 0 pi/3 2*pi/3 pi])
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','$-2\pi/3$','$-\pi/3$','$0$','$\pi/3$','$2\pi/3$','$\pi$'})
title('Spectral measure of rotation','interpreter','latex','fontsize',18)
xlabel('$\theta$','interpreter','latex','fontsize',18)
ylabel('$[K_{\epsilon}*\xi_g](\theta)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
box on
exportgraphics(gcf,'rotation_spec_meas.pdf','ContentType','vector','BackgroundColor','none')

%% compute operator folding matrices for T_2
[G,A2,~]=generate_matrices(x,y2,ker);
G=(G+G')/2;
cond(G)

%% compute spectral measures (using same g and theta as before)
specmeas2=unitary_spectral_measure(G,A2,g,theta,6,'equi',0.01);

%% plot spectral measures
%normalize
specmeas2=specmeas2/(pi*mean(specmeas2));
figure
plot(theta,specmeas2,'linewidth',2)
hold on
ylim([0.31 0.33])
xlim([-pi pi])
xticks([-pi -2*pi/3 -pi/3 0 pi/3 2*pi/3 pi])
title('Spectral measure of M\"obius map','interpreter','latex','fontsize',18)
xlabel('$\theta$','interpreter','latex','fontsize',18)
ylabel('$[K_{\epsilon}*\xi_g](\theta)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
box on
set(groot,'defaultAxesTickLabelInterpreter','latex');  
xticklabels({'$-\pi$','$-2\pi/3$','$-\pi/3$','$0$','$\pi/3$','$2\pi/3$','$\pi$'})

exportgraphics(gcf,'mobius_map_spec_meas.pdf','ContentType','vector','BackgroundColor','none')

%% get 'exact' value of rho_g(pi/3)
val_exact=unitary_spectral_measure(G,A2,g,theta0,6,'equi',0.0005);

%% convergence analysis for different m and epsilon at point pi/3
m_range=[1 2 3 4 5 6];
eps_range=10.^(-3:0.2:0);
theta0=pi/3;
vals=zeros(6,length(eps_range));
for i=1:6
    for j=1:length(eps_range)
        vals(i,j)=unitary_spectral_measure(G,A2,g,theta0,m_range(i),'equi',eps_range(j));
    end
end

%% plot figure showing convergence rates for different m as epsilon varies
figure
for i=1:6
    er=abs(vals(i,:)-val_exact)/abs(val_exact);
    loglog(eps_range,er,'linewidth',2)
    hold on
end
box on
ax=gca; ax.FontSize=14;
L=legend('$m=1$','$m=2$','$m=3$','$m=4$','$m=5$','$m=6$','interpreter','latex','location','best');
L.AutoUpdate='off';
title('$|[K_{\epsilon}*\xi_g](\pi/3)-\rho_g(\pi/3)]|/|\rho_g(\pi/3)|$','interpreter','latex','fontsize',18)
xlabel('$\epsilon$','interpreter','latex','fontsize',18)
ylim([10^(-15) 10^(0)])
txt=text(10^(-2.5),10^(-2.4),'$\mathcal{O}(\epsilon)$','interpreter','latex','fontsize',16); %show theoretical convergence rates on plot
p1=plot(10.^(-2.6:0.1:-1.9),10.^((-2.6:0.1:-1.9)-0.7),'--','color','black','linewidth',1);
txt2=text(10^(-1.8),10^(-11.9),'$\mathcal{O}(\epsilon^6)$','interpreter','latex','fontsize',16);
plot(10.^(-2.1:0.1:-1.4),10.^(6*(-2.1:0.1:-1.4)),'--','color','black','linewidth',1)
set(txt,'rotation',13)
set(txt2,'rotation',40)
exportgraphics(gcf,'mobius_map_error_comp.pdf','ContentType','vector','BackgroundColor','none')

%% define mobius maps and kernels

function F = mobius(z)
    F=exp(1i*pi/3)*z;
end

function F = mobius2(z)
    a=sqrt(2)*exp(1i*pi*sqrt(3));  %need |a|^2-|b|^2=1
    b=exp(1i*pi*9/7);
    F=(a*z+b)/(conj(b)*z+conj(a));     
end

function k = kernel(x,y,scale)
    d=2*atanh(abs((y-x)/(1-conj(x)*y)));
    k=exp(-scale*d^2);
end