%**************************************************************************
% This algorithm perform Levenberg-Marquardt non-linear curve fit
% of gauss amplitude function with offset y=y0+A*exp(-((x-xc)^2/(2*w^2)))
%**************************************************************************
clear all
close all
%**************************************************************************
% initialization parameters
%**************************************************************************
% initial amplitude
x01=1;
% initial center
x02=35;
% initial width
x03=1;
% initial offset
x04=0;
% initial parameters
x0=[x01 x02 x03 x04];
%**************************************************************************
% maximum number of iterations
k_max=1000;
% initialize control fitting parameters
% epsilon 1
epsilon_1=1.0e-8;
% epsilon 2
epsilon_2=1.0e-8;
% tau
tau=1.0e-3;
%**************************************************************************
% number of experimental points
M=100;
% initialize x (t) and y vectors of gauss function
t_data=zeros(1,M);
y_data=zeros(1,M);
% initialize amplitude
x1=107.8;
% initialize center
x2=37.4;
% initialize width
x3=15.7;
% initialize offset
x4=122.6;
% t_data vector
multiplicator_width=4;
t_data_start=x2-multiplicator_width*x3;
t_data_stop=x2+multiplicator_width*x3;
t_data=linspace(t_data_start,t_data_stop,M);
% y vector
y_data=x1*exp(((-1.0)*((t_data(1,:)-x2).^2)./(2*x3^2)))+x4;
% plot experimental curve
plot(t_data(1,:),y_data(1,:),'-o');
%**************************************************************************
% perform fit using levmar_gauss_amp function
%**************************************************************************
fit_result=levmar_gauss_amp(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau);
%**************************************************************************
% define function for levenberg-marquardt algorithm - for fit of gaussian
% function with amplitude
%**************************************************************************
% Explanation of function parameters
% parameters of function are: t_data (x coordinates of gaussian function);
% y_data (y coordinates of gaussian function); initial fit
% parameters e.g. x0 = [amplitude center width offset]; maximum number of iterations e.g.
% k_max=1000; epsilon_1=1.0e-8 (good value); epsilon_2=1.0e-8 (good value); 
% tau=1.0e-3 (good value);
%**************************************************************************
function [x_new] = levmar_gauss_amp(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau)
    % initialize basic fitting parameters
    % initialize iteration variable
    k=0;
    ni=2;
    % calculate number of points
    M=length(t_data(1,:));
    % initialize x vector
    x=x0;
    % initialize Jacobian matrix
    J=zeros(M,4);
    J(:,1)=(-1.0)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2));
    J(:,2)=(-1.0)*(x(1,1)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2)).*2.*(t_data(1,:)-x(1,2)))./(2*x(1,3)^2);
    J(:,3)=((-1.0)*x(1,1)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2)).*(t_data(1,:)-x(1,2)).^2)./x(1,3)^3;
    J(:,4)=-1;
    % calculate A matrix
    A=transpose(J(:,:))*J(:,:);
    % calculate f function
    f=y_data(1,:)-x(1,1)*exp((-1.0)*(t_data(1,:)-x(1,2)).^2./(2*x(1,3)^2))-x(1,4);
    % calculate g matrix
    g=transpose(J(:,:))*transpose(f(1,:));
    % calculate norm g
    g_norm=sqrt(sum(g(:,1).*g(:,1)));
    % calculate boolean parameter
    found_bool=g_norm < epsilon_1;
    % calculate mi parameter
    mi=tau*max(diag(A(:,:)));
    %**************************************************************************
    % main loop
    %**************************************************************************
    while ((~found_bool) & (k < k_max))
        % iterate
        k=k+1
        % calculate h_lm
        B=A(:,:)+mi*eye(4);
        C=inv(B(:,:));
        h_lm=(-1.0)*transpose(g(:,1))*C(:,:);
        % calculate norm h_lm
        h_lm_norm=sqrt(sum(h_lm(1,:).*h_lm(1,:)));
        % calculate norm x
        x_norm=sqrt(sum(x(1,:).*x(1,:)));
        if (h_lm_norm < epsilon_2*(x_norm+epsilon_2))
            found_bool=1;
        else
            x_new=x(1,:)+h_lm(1,:)
            % calculate f
            f=y_data(1,:)-x(1,1)*exp((-1.0)*(t_data(1,:)-x(1,2)).^2./(2*x(1,3)^2))-x(1,4);
            % calculate new f
            f_new=y_data(1,:)-x_new(1,1)*exp((-1.0)*(t_data(1,:)-x_new(1,2)).^2./(2*x_new(1,3)^2))-x_new(1,4);
            % calculate F
            F=0.5*sum(f(1,:).*f(1,:));
            % calculate new F
            F_new=0.5*sum(f_new(1,:).*f_new(1,:));
            % calculate ro numerator
            ro_numerator=F-F_new;
            % caculate ro denominator
            ro_denominator=0.5*h_lm(1,:)*(mi*transpose(h_lm(1,:))-g(:,1));
            % calculate ro
            ro=ro_numerator/ro_denominator;
            if (ro > 0)
                x=x_new;
                % calculate Jacobian matrix
                J=zeros(M,4);
                J(:,1)=(-1.0)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2));
                J(:,2)=(-1.0)*(x(1,1)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2)).*2.*(t_data(1,:)-x(1,2)))./(2*x(1,3)^2);
                J(:,3)=((-1.0)*x(1,1)*exp((-1.0)*((t_data(1,:)-x(1,2)).^2)./(2*x(1,3)^2)).*(t_data(1,:)-x(1,2)).^2)./x(1,3)^3;
                J(:,4)=-1;
                % calculate A matrix
                A=transpose(J(:,:))*J(:,:);
                % calculate f function
                f=y_data(1,:)-x(1,1)*exp((-1.0)*(t_data(1,:)-x(1,2)).^2./(2*x(1,3)^2))-x(1,4);
                % calculate g matrix
                g=transpose(J(:,:))*transpose(f(1,:));
                % calculate g norm
                g_norm=sqrt(sum(g(:,1).*g(:,1)));
                % calculate found boolean parameter
                found_bool=g_norm <= epsilon_1;
                mi=mi*max([(1/3) (1-(2*ro-1)^3)]);
                ni=2;
            else
                mi=mi*ni;
                ni=2*ni;
            end
        end
    end
end
%**************************************************************************
% end
%**************************************************************************