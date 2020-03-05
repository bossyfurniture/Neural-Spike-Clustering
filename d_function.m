%function for displaying discriminant function
%pass the pattern vector and covariances and means of a and b.
%Show the class centers with one pixel.
function [tz,gz]=d_function(z,sa,sb,a,b)
[m,n]=size(z);
%set up the mean vectors.
r=ones(m,2);
ua=[a(1)*r(:,1) a(2)*r(:,2)];
ub=[b(1)*r(:,1) b(2)*r(:,2)];

%form the pieces of the discriminant function.
%tz=log(det(sa)/det(sb))-1;
tz=0.5*log(det(sa)/det(sb));
tz=tz+0.5*((z-ub)/sb*(z-ub)'-(z-ua)/sa*(z-ua)');
%tz is the discriminant function for the range of x and y.
%find coordinates where it is greater than zero.
gz=(tz>0);
%color the class mean pixels differently from background color.
gz(11,11)= ~gz(11,11);
gz(31,31)= ~gz(31,31);
%plot the centers and discriminant boundary.
imagesc(z(:,1),z(:,2),gz);axis('xy')
xlabel('z1')
ylabel('z2')
%compute the normalized covariance of a and b as t12,t21.
t12=sa(1,2)/sa(1,1);
t21=sb(1,2)/sb(1,1);
%get the variance of a and b.
t1=sa(1,1);
t2=sb(1,1);
tit=strcat(['Var of a =' num2str(t1) ';Var of b=' num2str(t2) ',Covar of a=' num2str(t12) ';Covar of b=' num2str(t21)]);
title(tit)



