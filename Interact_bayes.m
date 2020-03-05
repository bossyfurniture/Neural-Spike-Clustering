%intermediate script to set up sliders.
%  Top slider is covariance
% and bottom slider is variance ratio.
sc=get(slide,'Value')
mul=get(slide2,'Value')
sa=[1 sc
   sc 1];
sb=[1 -sc 
    -sc 1];
sa=(2^mul)*sa;
sb=(2^(-mul))*sb;
%call discriminant function using values from sliders.
[tz,gz]=d_function(z,sa,sb,a,b);
