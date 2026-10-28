
function [arrayt1, arrayt2] = moveplate(arrayt1, arrayt2, Q, width);
%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%moveplate determines the position of the target points at time t. 
% The plate is moved with amplitude amp. 
%-----------------------------------------------------------------  

in_fluid;

global time;

amp = width/8;

for i=1:Q,
    arrayt1(i,1)= width/2 + amp*sin(2*pi*time/time_final);
    arrayt2(i,1)= (width/4) + (i-1)*ds;
end
