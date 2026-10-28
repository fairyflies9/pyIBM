function [array]=vorticity(array);
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%vorticity.m is a function that will calculate the vorticity of of a 2D
%fluid field. The input is a matrix to hold the values of voriticity.
%Global variables include the x and y-components of the fluid velocity.

in_fluid;  %read in the fluid parameters
global u v; %define global parameters

%Calculate the vorticity in the middle of the plot
for i=2:(M-1),  
    for j=2:(N-1),  
      array(i,j) = (v(i+1,j)-v(i-1,j))/(2*dx) - (u(i,j+1)-u(i,j-1))/(2*dx); 
    end
end
 
%Hand the points on the edges of the domain
for j=2:N-1,
    vort(1,j)=(v(2,j)-v(M,j))/(2*dx) - (u(i,j+1)-u(i,j-1))/(2*dx); 
    vort(M,j)=(v(1,j)-v(M-1,j))/(2*dx) - (u(i,j+1)-u(i,j-1))/(2*dx); 
end

for i=2:M-1,
    vort(i,1)=(v(i+1,j)-v(i-1,j))/(2*dx) - (u(i,2)-u(i,N))/(2*dx); 
    vort(i,N)=(v(i+1,j)-v(i-1,j))/(2*dx) - (u(i,1)-u(i,N-1))/(2*dx); 
end

    vort(1,1)=(v(2,1)-v(M,1))/(2*dx) - (u(1,2)-u(1,N))/(2*dx); 
    vort(1,N)=(v(2,N)-v(M,N))/(2*dx) - (u(1,1)-u(1,N-1))/(2*dx); 
    
    vort(M,1)=(v(1,1)-v(M-1,1))/(2*dx) - (u(M,2)-u(M,N))/(2*dx); 
    vort(M,N)=(v(1,N)-v(M-1,N))/(2*dx) - (u(M,1)-u(M,N-1))/(2*dx); 