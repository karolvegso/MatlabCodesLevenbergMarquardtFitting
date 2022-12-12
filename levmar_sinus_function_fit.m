%**************************************************************************
% This algorithm solves non-linear curve fit of sinusoidal function 
% based on levenberg-marquardt approach
%**************************************************************************
clear all
close all
%**************************************************************************
% initial parameters
% amplitude
x01=0;
% phase shift
x02=0;
% offset
x03=0;
% initial vector
x0=[x01 x02 x03];
%**************************************************************************
% maximum number of iterations
k_max=1000;
% epsilon 1
epsilon_1=1.0e-8;
% epsilon 2
epsilon_2=1.0e-8;
% tau
tau=1.0e-3;
%**************************************************************************
% number of experimental points
M=100;
% initialize x and y vectors
t_data=zeros(1,M);
y_data=zeros(1,M);
% generate x points
for index_0=1:M
    t_data(1,index_0)=((2*pi)/M)*(index_0-1)
end
%**************************************************************************
% set parameters of sinusoidal function
% amplitude
x1=123;
% phase shift
x2=pi/4;
% offset
x3=17;
% generate y ponits
y_data(1,:)=x1*sin(t_data(1,:)+x2)+x3;
%**************************************************************************
% plot experimental function
plot(t_data(1,:), y_data(1,:), '-o');
%**************************************************************************
% perform fit using levmar_sinus function
%**************************************************************************
fit_result=levmar_sinus(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau);
%**************************************************************************
% define function for levenberg-marquardt algorithm - for fit of sinusoidal
% function
%**************************************************************************
% Explanation of function parameters
% parameters of function are: t_data (x coordinates of sinusoidal
% function); y_data (y coordinates of sinusoidal function); initial fit
% parameters e.g. x0 = [0 0 0]; maximum number of iterations e.g.
% k_max=1000; epsilon_1=1.0e-8 (good value); epsilon_2=1.0e-8 (good value); 
% tau=1.0e-3 (good value);
%**************************************************************************
function [x_new] = levmar_sinus(t_data, y_data, x0, k_max, epsilon_1, epsilon_2, tau)
    %**************************************************************************
    % beginning of algorithm
    %**************************************************************************
    % initial iteration variable
    k=0;
    ni=2;
    M=length(t_data(1,:));
    % define Jacobian matrix
    J=zeros(M,3);
    % fill Jacobian matrix
    J(:,1)=(-1.0)*sin(t_data(1,:)+x0(1,2));
    J(:,2)=(-1.0)*x0(1,1)*cos(t_data(1,:)+x0(1,2));
    J(:,3)=-1;
    % calculate A matrix
    A(:,:)=transpose(J(:,:))*J(:,:);
    % caclulate function f
    f=zeros(1,M);
    f=y_data(1,:)-x0(1,1)*sin(t_data(1,:)+x0(1,2))-x0(1,3);
    % calculate g
    g=transpose(J(:,:))*transpose(f(1,:));
    % calculate g norm
    g_norm=sqrt(sum(g(:,1).*g(:,1)));
    % boolean variable
    found_bool=(g_norm <= epsilon_1);
    % define mi
    mi=tau*max(diag(A(:,:)));
    % initialize x vector
    x=x0(1,:);
    while ((~found_bool) & (k < k_max))
        % increase iteration by one
        k=k+1
        B=A+mi*eye(3);
        h_lm=(-1.0)*transpose(g)*inv(B);
        h_lm_norm=sqrt(sum(h_lm(1,:).*h_lm(1,:)));
        x_norm=sqrt(sum(x(1,:).*x(1,:)));
        if (h_lm_norm <= epsilon_2*(x_norm+epsilon_2))
            found_bool=1;
        else
            x_new=x(1,:)+h_lm(1,:)
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
            ro_denominator=0.5*h_lm(1,:)*(mi*transpose(h_lm(1,:))-g(:,1));
            % calculate ro - gain ratio
            ro=(F_x-F_x_new)/ro_denominator;
            if (ro > 0)
                x=x_new(1,:);
                % define Jacobian matrix
                J=zeros(M,3);
                % fill Jacobian matrix
                J(:,1)=(-1.0)*sin(t_data(1,:)+x(1,2));
                J(:,2)=(-1.0)*x(1,1)*cos(t_data(1,:)+x(1,2));
                J(:,3)=-1;
                % calculate A matrix
                A(:,:)=transpose(J(:,:))*J(:,:);
                % caclulate function f
                f=zeros(1,M);
                f=y_data(1,:)-x(1,1)*sin(t_data(1,:)+x(1,2))-x(1,3);
                % calculate g
                g=transpose(J(:,:))*transpose(f(1,:));
                % calculate g norm
                g_norm=sqrt(sum(g(:,1).*g(:,1)));
                % boolean variable
                found_bool=(g_norm <= epsilon_1);
                % calculate mi
                mi=mi*max([(1/3) (1-(2*ro-1)^3)]);
                % define ni
                ni=2;
            else
                % calculate mi
                mi=mi*ni;
                % calculate ni
                ni=2*ni;
            end
        end    
    end
    %**************************************************************************
    % end of algorithm 
    %**************************************************************************
end
%**************************************************************************
% End
%**************************************************************************