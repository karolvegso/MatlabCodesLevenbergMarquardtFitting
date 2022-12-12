%**************************************************************************
% This algorithm perform Levenberg-Marquardt non-linear curve fit
% of Voigt function given by convolution of Lorentz function and Gaussian
% function
%**************************************************************************
clear all
close all
%**************************************************************************
% initial parameters
%**************************************************************************
% initial offset
x01=0;
% initial area
x02=1000;
% initial width - Lorentz
x03=1;
% initial width - Gauss
x04=1;
% initial center
x05=40;
% initial parameters
x0=[x01 x02 x03 x04 x05];
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
% initialize x (t_data) and y vectors of Voigt function
t_data=zeros(1,M);
y_data=zeros(1,M);
% initialize offset
x1=15.3;
% initialize area
x2=102.6;
% initialize width - Lorentz
x3=5.1;
% initialize width - Gauss
x4=2.6;
% initialize center
x5=42.7;
% fill x vector - Voigt function parameters
x_Voigt_fun=[x1 x2 x3 x4 x5];
% t_data vector
multiplicator_width=4;
t_data_start=x5-multiplicator_width*(x3+x4);
t_data_stop=x5+multiplicator_width*(x3+x4);
t_data=linspace(t_data_start,t_data_stop,M);
% y_data vector
% define Voigt function
number_01=(2*log(2))/pi^(3/2);
number_02=sqrt(log(2));
number_03=sqrt(4*log(2));
Voigt_fun=@(p,r) p(1)+number_01*p(2)*(p(3)/p(4)^2)*...
    integral(@(s) exp(-s^2)./((number_02*(p(3)/p(4)))^2+(number_03*((r-p(5))/p(4))-s).^2), -Inf, Inf, 'ArrayValued', true);
y_data=Voigt_fun(x_Voigt_fun(1,:), t_data(1,:));
% plot experimental curve
plot(t_data(1,:),y_data(1,:),'-o');
%**************************************************************************
% perform fit using levmar_gauss_area function
%**************************************************************************
fit_result=levmar_voigt_area(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau);
%**************************************************************************
% define function for levenberg-marquardt algorithm - for fit of Voigt
% function with area, Voigt function is given as convolution of Lorentz and
% Gaussian functions
%**************************************************************************
% Explanation of function parameters
% parameters of function are: t_data (x coordinates of Voigt function);
% y_data (y coordinates of Voigt function); initial fit
% parameters e.g. x0 = [offset area width_Lorentz width_Gauss center]; 
% maximum number of iterations e.g. k_max=1000; 
% epsilon_1=1.0e-8 (good value); epsilon_2=1.0e-8 (good value); 
% tau=1.0e-3 (good value);
%**************************************************************************
function [x_new] = levmar_voigt_area(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau)
    % definition of Voigt function at the beginning
    % define Voigt function
    number_01=(2*log(2))/pi^(3/2);
    number_02=sqrt(log(2));
    number_03=sqrt(4*log(2));
    Voigt_fun=@(p,r) p(1)+number_01*p(2)*(p(3)/p(4)^2)*...
        integral(@(s) exp(-s^2)./((number_02*(p(3)/p(4)))^2+(number_03*((r-p(5))/p(4))-s).^2), -Inf, Inf, 'ArrayValued', true);
    % initialize basic fitting parameters
    % initialize iteration variable
    k=0;
    ni=2;
    % calculate number of points
    M=length(t_data(1,:));
    % initialize x vector
    x=x0;
    % initialize Jacobian matrix
    J=zeros(M,5);
    % difference
    h=1.0e-4;
    % Jacobian for offset
    J(:,1)=-1;
    % Jacobian for area
    x_h=x(1,:);
    x_h(1,2)=x(1,2)+h;
    J(:,2)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
    % Jacobian for width - Lorentz
    x_h=x(1,:);
    x_h(1,3)=x(1,3)+h;
    J(:,3)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
    % Jacobian for width - Gauss
    x_h=x(1,:);
    x_h(1,4)=x(1,4)+h;
    J(:,4)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
    % Jacobian for center
    x_h=x(1,:);
    x_h(1,5)=x(1,5)+h;
    J(:,5)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
    % calculate A matrix
    A=transpose(J(:,:))*J(:,:);
    % calculate f function
    f=y_data(1,:)-Voigt_fun(x(1,:), t_data(1,:));
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
        k=k+1
        % calculate h_lm
        B=A(:,:)+mi*eye(5);
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
            f=y_data(1,:)-Voigt_fun(x(1,:), t_data(1,:));
            % calculate new f
            f_new=y_data(1,:)-Voigt_fun(x_new(1,:), t_data(1,:));
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
                % initialize Jacobian matrix
                J=zeros(M,5);
                % difference
                h=1.0e-4;
                % Jacobian for offset
                J(:,1)=-1;
                % Jacobian for area
                x_h=x(1,:);
                x_h(1,2)=x(1,2)+h;
                J(:,2)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
                % Jacobian for width - Lorentz
                x_h=x(1,:);
                x_h(1,3)=x(1,3)+h;
                J(:,3)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
                % Jacobian for width - Gauss
                x_h=x(1,:);
                x_h(1,4)=x(1,4)+h;
                J(:,4)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
                % Jacobian for center
                x_h=x(1,:);
                x_h(1,5)=x(1,5)+h;
                J(:,5)=(-Voigt_fun(x_h(1,:), t_data(1,:))+Voigt_fun(x(1,:), t_data(1,:)))/h;
                % calculate A matrix
                A=transpose(J(:,:))*J(:,:);
                % calculate f function
                f=y_data(1,:)-Voigt_fun(x(1,:), t_data(1,:));
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
% end of program
%**************************************************************************