function moving_plate;
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%Moving plate is the main program that will be used to simulate the 
%immersed boundary method in 2D with a plate attached to a "target plate"
%that moves with a sinusoidal motion. 
%-----------------------------------------------------------------  

clear all %clear all variables
clf       %and figures

in_fluid; %parameters for fluid solver

global u v f1 f2 press w1 w2 uft vft w1ft w2ft pft time;
global a b c d theta_1 theta_2;

%Set the stiffness of the springs between the actual boundary and the
%target boundary
stiff = 1;
width = 0.4;
ptime = 0;  %keeps track of when to graph
T = round(time_final/dt); %number of timesteps
time = 0;     %current time, set initially to zero
k=1;    %keeps track of frame number for movie
Q = M;  %sets the number of boundary points

%--------------------------------------------------------------------------
%initialize vectors and matrices
u = zeros(M, N);    %velocity in x-direction
v = zeros(M, N);    %velocity in y-direction
f1 = zeros(M, N);    %force in x-direction
f2 = zeros(M, N); 
press = zeros(M, N);    %pressure
t = zeros(1, T);        %time
mag = zeros(M, N);      %velocity magnitudes

%initialize matrices   
w1 = zeros(M, N);  
w2 = zeros(M, N); 

uft = zeros(M, N);  %fourier transforms of the velocities
vft = zeros(M,N);
pft = zeros(M,N);   %fourier transform of the pressure
xcoor = zeros(M,N);   %holds the x and y Cartesian coordiante for the (i,j) gridpoint
ycoor = zeros(M,N); 
vort = zeros(M,N);    %holds the values of the vorticity

%Initialize vectors that hold boundary locations, target locations, and
%forces.
b1 = zeros(Q,1);
b2 = zeros(Q,1);
bt1 = zeros(Q,1);
bt2 = zeros(Q,1);

fb1 = zeros(Q,1);
fb2 = zeros(Q,1);

%holds the cartesian coordinates of each grid point for plotting
for i=1:M,
    for j=1:N,
        xcoor(i,j)=(i-1)*dx;
        ycoor(i,j)=(j-1)*dx;
    end
end

%------------------------------------------------------------------------------------    
%Set values for matrix used in fluid solver so they are only calculated once.
%These values are used to calculate u_hat and p_hat in equations 14.51 and
%14.52 in Peskin and McQueen, 1996.
for i=1:M,  
   for j = 1:N, 
      if(or(and((i==1),(j==1)),and((i==1), (j==(N/2)+1)))),
        a(i,j)=0;  
        b(i,j)=0;  
        c(i,j)=0;  
      elseif(or(and((i==(M/2)+1),(j==1)),and((i==(M/2)+1),(j==(N/2)+1)))),  
        a(i,j)=0;  
        b(i,j)=0;  
        c(i,j)=0;  
      else
        a(i,j)= sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
        b(i,j)= sin(2*pi*(i-1)/M)*sin(2*pi*(j-1)/N)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
        c(i,j)= sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)/((sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M)+sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N)));  
      end  
   end
end



for i=1:M,  
    for j = 1:N,  
        d(i,j)=1/(1+(4*nu*dt/(dx*dx))*(sin(pi*(i-1)/M)*sin(pi*(i-1)/M)+sin(pi*(j-1)/N)*sin(pi*(j-1)/N)));  
    end
end

%--------------------------------------------------------------------------

%calculate the initial velocity
%just let the initial velocity be zero  
  
%--------------------------------------------------------------------------
%set the initial boundary location and initial target location
for i=1:Q,
    b1(i,1)=width/2;
    b2(i,1)=(width/4) + (i-1)*ds;
end

for i=1:Q,
    bt1(i,1)=b1(i,1);
    bt2(i,1)=b2(i,1);
end


%--------------------------------------------------------------------------

%create a movie object and name it
%aviobj = avifile('fluid_movie_16', 'fps', 3);
%--------------------------------------------------------------------------

%step through time, calculate u, v and press at each time step
while time<time_final,
    %zero the forces f1 and f2
    f1 = zeros(M,N);
    f2 = zeros(M,N);
    
    %move the target points for the plate
    [bt1, bt2] = moveplate(bt1, bt2, Q, width);
    
    %calculate the force each boundary point applies to the fluid
    [fb1, fb2] = target_force(b1, b2, bt1, bt2, fb1, fb2, stiff, Q);
    
    %spread the force to the fluid grid
    forcespread(b1, b2, fb1, fb2, Q);
    
    %calculate new velocity and pressure
    fluid_solve;
    
    %move the boundary at the local velocity
    [b1, b2] = move_boundary(b1, b2, Q, width);
    
    %--------------------------------------------------
    %plot results every dptime
    if time>ptime,

        %Make a voriticy plot with the actual wing and boundary wing
        %superimposed.
        figure(1)
        hold off
        [vort]=vorticity(vort);  %calculate the vorticity
        
        %make a color plot of the magnitude of the velocity
        pcolor(xcoor, ycoor, vort);    
        %caxis([0 .1]);
        shading interp;
        
        hold on
        %plot the target and boundary points
        plot(b1, b2, 'k.');
        plot(bt1, bt2, 'r.');
        axis([0 (0.4-dx) 0 (0.4-dx)]);
        
        ptime = dptime + ptime; %update ptime
        F(k) = getframe;    %add frame to the movie
        frame = getframe;
        %aviobj = addframe(aviobj, frame);
        k = k+1;    %update frame number
    end
    %-----------------------------------------------------
     

    time = time+dt; %update time
end

%close movie
%aviobj = close(aviobj);

