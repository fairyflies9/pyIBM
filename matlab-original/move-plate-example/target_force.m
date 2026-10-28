function [force1, force2] = target_force(array1, array2, arrayt1, arrayt2, force1, force2, stiff, Q);

%----------------------------------------------------------------
%Written by Laura Miller, 02/12/08. Warning: This code is far from
%optimized or pretty :) Use at your own risk
%
%targetforce determines the force the boundary applies to the fluid as a results
%of being "tethered" to target points.
%Please see Miller and Peskin and McQueen, 2004 for more details.
%
%Inputs
%array1 and array2 hold the cartesian coordinates of the boundary points.
%arrayt1 and arrayt2 hold the Cartesian coordinates of the target points.
%force1 and force1 will hold the force each boundary point applies to the
%fluid.
%Q is the number of boundary points, or the length of the above arrays.
%stiff is the stiffness of the springs that hold each boundary point to 
%a target point.
%
%Matrices modified:
%force1 and force2 are updated
%-----------------------------------------------------------------  

%Loop through the boundary points and calculate the force
for i = 1:Q,
    force1(i,1) = stiff*(arrayt1(i,1)-array1(i,1));
    force2(i,1) = stiff*(arrayt2(i,1)-array2(i,1));
end