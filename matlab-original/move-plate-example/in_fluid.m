%filename in_fluid.m
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk. It was written for clarity in
%learning rather than employing the practices of good code writing or
%speed.
%
%in_fluid holds the parameter values used in the 2D fluid solver.
%----------------------------------------------------------------- 

%spatial grid size
%For this version of the code, M should equal N, and length should equal
%width
N = 16;
M = 16;

%Mechanical variables
muu = 3.0;  %dynamic viscosity
p= 2.0;    %density
nu = muu/p; %kinematic viscosity

%Spatial variables
length = 0.4;  %length of domain
%width = 0.4;   %width of domain
dx = length/N;        %absolute length of grid  
ds = length/(2*N);   %boundary step  
dt = dx*dx/nu;      %time step  

%new timing parameter values
time_final = 10; 	%total time
L = 20; %number of times to graph velocity and pressure
dptime = time_final/L;   %graphs every dptime  