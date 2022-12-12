%**************************************************************************
% This algorithm performs Powell's dog leg method fitting on sinus function
% author: Karol Vegso
% affiliation: Institute of Physics, Slovak Academy of Sciences
%**************************************************************************
clear all
close all
clc
%**************************************************************************
% generate experimental sinusoidal function
% number of experimental points
M=100;
% parameters of sinusoidal function
% amplitude
a=125.6;
% phase
b=pi/3;
% offset
c=56.7;
% x step
t_step=(2*pi)/M;
% generate x coordinate
t_data=0:t_step:(M-1)*t_step;
% generate y coordinate
y_data=a*sin(t_data(1,:)+b)+c;
% plot experimental function
plot(t_data(1,:),y_data(1,:),'-o');
%**************************************************************************
% initial fitting parameters - amplitude, phase, offset
%**************************************************************************
% initial amplitude
x01=1;
% initial phase
x02=0;
% initial offset
x03=0;
% initial parameters
x0=[x01 x02 x03];
%**************************************************************************
% fitting parameters
%**************************************************************************
% maximum iterations
k_max=1000;
% epsilon 1
epsilon_1=1.0e-15;
% epsilon 2
epsilon_2=1.0e-15;
% epsilon 3
epsilon_3=1.0e-20;
% initial delta parameter
delta=1;
%**************************************************************************
% fit simulated curve
fit_result=powells_dog_leg_sinus(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, epsilon_3, delta);
%**************************************************************************
% function to perform Powell's dog leg fitting method on sinusoidal
% function
% input aparmeters are t_data (x values), y_data (y values), x0 (initial
% fitting parameters - amplitude, phase, offset), k_max (maximum number of
% iterations), epsilon_1 (good value 1.0e-15), epsilon_2 (good value
% 1.0e-15), epsilon_3 (good value 1.0e-20), delta (good initial value 1)
%**************************************************************************
function [x_new] = powells_dog_leg_sinus(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, epsilon_3, delta)
    %**************************************************************************
    % begin of function
    %**************************************************************************
    % number of simulated or experimental points
    M=length(t_data(1,:));
    % initial iteration parameter
    k=0;
    % initialize fitting parameters to start with
    x=x0;
    % Jacobian matrix
    J=zeros(M,3);
    % caclulate derivation according to amplitude
    % fill first column of Jacobian matrix
    J(:,1)=(-1.0)*sin(t_data(1,:)+x(1,2));
    % calculate derivation according to phase
    % fill second column of Jacobian matrix
    J(:,2)=(-1.0)*x(1,1)*cos(t_data(1,:)+x(1,2));
    % calculate derivation according to offset
    % fill third column of Jacobian matrix
    J(:,3)=-1;
    % calculate f
    f=zeros(1,M);
    f=y_data(1,:)-x(1,1)*sin(t_data(1,:)+x(1,2))-x(1,3);
    % calculate g
    g=transpose(J(:,:))*transpose(f(1,:));
    % calculate norm f
    f_norm=sqrt(sum(f(1,:).*f(1,:)));
    % calculate norm g
    g_norm=sqrt(sum(g(:,1).*g(:,1)));
    % calculate boolean variable found
    found_bool=(f_norm <= epsilon_3) | (g_norm <= epsilon_1);
    %**************************************************************************
    % main loop
    %**************************************************************************
    while ((~found_bool) & (k < k_max))
        % increase iteration variable k
        k=k+1
        % Jacobian matrix
        J=zeros(M,3);
        % caclulate derivation according to amplitude
        % fill first column of Jacobian matrix
        J(:,1)=(-1.0)*sin(t_data(1,:)+x(1,2));
        % calculate derivation according to phase
        % fill second column of Jacobian matrix
        J(:,2)=(-1.0)*x(1,1)*cos(t_data(1,:)+x(1,2));
        % calculate derivation according to offset
        % fill third column of Jacobian matrix
        J(:,3)=-1;
        % calculate f
        f=zeros(1,M);
        f=y_data(1,:)-x(1,1)*sin(t_data(1,:)+x(1,2))-x(1,3);
        % calculate g
        g=transpose(J(:,:))*transpose(f(1,:));
        % calculate norm g
        g_norm=sqrt(sum(g(:,1).*g(:,1)));
        % calculate J(x)*g(x)
        Jg=J(:,:)*g(:,1);
        % calculate norm Jg
        Jg_norm=sqrt(sum(Jg(:,1).*Jg(:,1)));
        % calculate alpha
        alpha=g_norm^2/Jg_norm^2;
        % calculate h_sd
        h_sd=(-1.0)*alpha*g(:,1);
        h_sd=transpose(h_sd);
        % calculate h_gn
        A=transpose(J(:,:))*J(:,:);
        B=transpose(J(:,:))*transpose(f(1,:));
        h_gn=(-1.0)*transpose(B)*inv(A);
        % calculate h_dl
        % calculate norm h_gn
        h_gn_norm=sqrt(sum(h_gn(1,:).*h_gn(1,:)));
        % calculate norm alpha*h_sd
        alpha_h_sd=alpha*h_sd(1,:);
        alpha_h_sd_norm=sqrt(sum(alpha_h_sd(1,:).*alpha_h_sd(1,:)));
        % calculate norm h_sd
        h_sd_norm=sqrt(sum(h_sd(1,:).*h_sd(1,:)));
        % caclulate h_dl
        if (h_gn_norm <= delta)
            h_dl=h_gn(1,:);
        elseif (alpha_h_sd_norm >= delta)
            h_dl=(delta/h_sd_norm)*h_sd(1,:);
        else
            beta=sym('beta');
            h_dl=alpha*h_sd(1,:)+beta*(h_gn(1,:)-alpha*h_sd(1,:));
            h_dl_norm=sqrt(sum(h_dl(1,:).*h_dl(1,:)));
            eqn=(h_dl_norm == delta);
            beta_result=solve(eqn, beta, 'Real', true);
            beta_result=double(beta_result);
            clear beta;
            h_dl=alpha*h_sd(1,:)+beta_result(2,1)*(h_gn(1,:)-alpha*h_sd(1,:));
        end
        % calculate norm h_dl
        h_dl_norm=sqrt(sum(h_dl(1,:).*h_dl(1,:)));
        % calculate norm x
        x_norm=sqrt(sum(x(1,:).*x(1,:)));
        if (h_dl_norm <= epsilon_2*(x_norm+epsilon_2))
            found_bool=1;
        else
            x_new=x(1,:)+h_dl(1,:)
            % calculate F(x)
            % caclulate function f
            f=zeros(1,M);
            f=y_data(1,:)-x(1,1)*sin(t_data(1,:)+x(1,2))-x(1,3);
            F_x=0.5*sum(f(1,:).*f(1,:));
            % calculate F(x_new)
            f=zeros(1,M);
            f=y_data(1,:)-x_new(1,1)*sin(t_data(1,:)+x_new(1,2))-x_new(1,3);
            F_x_new=0.5*sum(f(1,:).*f(1,:));
            % ro denominator
            ro_denominator=0.5*h_dl(1,:)*(alpha*transpose(h_dl(1,:))-g(:,1));
            % calculate ro - gain ratio
            ro=(F_x-F_x_new)/ro_denominator;
            if (ro > 0)
                x=x_new(1,:);
                % Jacobian matrix
                J=zeros(M,3);
                % caclulate derivation according to amplitude
                % fill first column of Jacobian matrix
                J(:,1)=(-1.0)*sin(t_data(1,:)+x(1,2));
                % calculate derivation according to phase
                % fill second column of Jacobian matrix
                J(:,2)=(-1.0)*x(1,1)*cos(t_data(1,:)+x(1,2));
                % calculate derivation according to offset
                % fill third column of Jacobian matrix
                J(:,3)=-1;
                % calculate f
                f=zeros(1,M);
                f=y_data(1,:)-x(1,1)*sin(t_data(1,:)+x(1,2))-x(1,3);
                % calculate norm f
                f_norm=sqrt(sum(f(1,:).*f(1,:)));
                % calculate g
                g=transpose(J(:,:))*transpose(f(1,:));
                % calculate norm g
                g_norm=sqrt(sum(g(:,1).*g(:,1)));
                % cacluate boolean variable found
                found_bool=(f_norm <= epsilon_3) | (g_norm <= epsilon_1);
            end
            if (ro > 0.75)
                % calculate norm h_dl
                h_dl_norm=sqrt(sum(h_dl(1,:).*h_dl(1,:)));
                % calculate new delta
                delta=max([delta 3*h_dl_norm]);
            elseif (ro < 0.25)
                % calculate new delta
                delta=delta/2;
                % calculate norm x
                x_norm=sqrt(sum(x(1,:).*x(1,:)));
                % calculate boolean variable found
                found_bool=(delta <= epsilon_2*(x_norm+epsilon_2));
            end
        end
    end
    %**************************************************************************
    % end of function
    %**************************************************************************
end
%**************************************************************************
% end of program
%**************************************************************************