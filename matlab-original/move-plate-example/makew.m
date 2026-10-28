
function makew;  
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk. It was written for clarity in
%learning rather than employing the practices of good code writing or
%speed.
%
%makew goes with fluid_solve, which is used to solve the 2D Navier-Stokes 
%equations using a FFT method. The fluid domain should be a periodic 
%square. This code is based on Peskin and McQueen, 1996, and is described 
%in detail there.
%
%Global variables:
%u, v are the horizontal and vertical components of the fluid velocity
%press is the pressure
%f1 and f2 are the horizontal and vertical components of the applied force
%dx and dt are the spatial and time steps
%p is the fluid density
%M and N are the number of spatial steps in the x- and y-directions
%w1 and w2 are used to calculate the new u and v values.
%
%Matrices that are modified:
%w1 and w2
%----------------------------------------------------------------- 

in_fluid; %parameter values for the fluid solver.

global u v f1 f2 w1 w2;

%calculate some constants to begin
d_dx = 1.0/dx;  
dt_p=dt/p; 


for i=1:M,
    for j=1:N,
        %if we are on the edge of the boundary, then special values of i+1,
        %i=1, j+1, and j-1 should be calculated.
        jp1 = j+1;
        ip1=i+1;
        i_1 = i-1;
        j_1 = j-1;
        if(i==(M)),
            ip1 = 1;
        end
        if(i==1);
            i_1 = M;
        end
        if(j==(N)),
            jp1 = 1;
        end
        if(j==1),
            j_1=N;
        end
        %Calculate the values of w1 and w2 using u, v, f1 and f2
        %Determine whether or not upwind or downwind differencing is to be
        %used, then calculate the partial derivatives of u and v
        %accordingly. (Peskin and McQueen, equation 14.34).
        if(u(i,j)<0),
            du1 = (1/dx)*(u(ip1,j) - u(i,j));
            dv1 = (1/dx)*(v(ip1,j) - v(i,j));
        else
            du1 = (1/dx)*(u(i,j) - u(i_1,j));
            dv1 = (1/dx)*(v(i,j) - v(i_1,j));  
        end
        if(v(i,j)<0),
            du2 = (1/dx)*(u(i,jp1) - u(i,j));
            dv2 = (1/dx)*(v(i,jp1) - v(i,j));
        else
            du2 = (1/dx)*(u(i,j) - u(i,j_1));
            dv2 = (1/dx)*(v(i,j) - v(i,j_1)); 
        end
        %Calculate the values of w1 and w2 (Peskin and McQueen, equation
        %14.42).
         w1(i,j) = u(i,j)-dt*(u(i,j)*du1+v(i,j)*du2)+dt_p*f1(i,j);
         w2(i,j) = v(i,j)-dt*(u(i,j)*dv1+v(i,j)*dv2)+dt_p*f2(i,j);
    end
end