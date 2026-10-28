

function fluid_solve;
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%fluid_solve solves the 2D Navier-Stokes equations using a FFT method. The fluid
%domain should be a periodic square. This code is based on Peskin and
%McQueen, 1996, and is described in detail there
%
%Global variables:
%u, v are the horizontal and vertical components of the fluid velocity
%press is the pressure
%f1 and f2 are the horizontal and vertical components of the applied force
%a, b, c, and d are matrices that hold constants for calculating the new
%velocities
%dx and dt are the spatial and time steps
%p is the fluid density
%M and N are the number of spatial steps in the x- and y-directions
%
%Matrices modified:
%u and v are the new velocities
%press is the new pressure
%-----------------------------------------------------------------  

in_fluid;  %parameter values for the fluid solver


global u v f1 f2 press w1 w2 uft vft w1ft w2ft pft time;
global a b c d;


%calculate w1, w2, and take the Fourier transform of it. This will be used
%to caculate new velocities
%makew calculates w using equation 14.43 from Peskin and McQueen, 1996.
makew;  

w1ft = fft2(double(w1));  %fourier transform of w1, w2
w2ft = fft2(double(w2));


%calculate the transform of the velocity
%The velocity is calculated using equation 14.51 from Peskin and McQueen,
%1996.
for i=1:M,
    for j=1:N
        uft(i,j) = (w1ft(i,j)-(a(i,j)*w1ft(i,j)+b(i,j)*w2ft(i,j)))*d(i,j);   
        vft(i,j) = (w2ft(i,j)-(b(i,j)*w1ft(i,j)+c(i,j)*w2ft(i,j)))*d(i,j); 
    end
end


%---------------------------------------------------------------
%calculate the transform of the pressure
%Pressue is calculated using equation 14.52 from Peskin and McQueen, 1996.
 for i=1:M,
     for j=1:N,
         if(or(and((i==1),(j==1)),and((i==1), (j==(N/2)+1)))),
             pft(i,j)=0;
         elseif(or(and((i==(M/2)+1),(j==1)),and((i==(M/2)+1),(j==(N/2)+1)))), 
             pft(i,j)=0;
         else
             pft(i,j) = ((p*dx/dt)*(sin(2*pi*(i-1)/M)*imag(w1ft(i,j)) + sin(2*pi*(j-1)/N)*imag(w2ft(i,j))))/(sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M) + sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N));
             pft(i,j) = pft(i,j)-sqrt(-1)*((p*dx/dt)*(sin(2*pi*(i-1)/M)*real(w1ft(i,j)) + sin(2*pi*(j-1)/N)*real(w2ft(i,j))))/(sin(2*pi*(i-1)/M)*sin(2*pi*(i-1)/M) + sin(2*pi*(j-1)/N)*sin(2*pi*(j-1)/N));
         end
     end
 end

%-----------------------------------------------
  
%Take the inverse transforms of u, v, and press   
u = ifft2(double(uft));
v = ifft2(double(vft));
press = ifft2(double(pft));

%get rid of the imaginary part of u, v, press (should be only round-off
%error)
u = real(u);
v = real(v);
press = real(press);
  
