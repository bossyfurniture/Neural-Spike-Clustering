%This is the main program to demonstrate Bayes classification
%Set range of x and y, the two elements of the pattern vector
figure;
x=-2:.1:2;
y=x;
%Form the pattern vector.  
z=[x' y'];
%Set the class means. (1,1), (-1,-1)
b=[1 1];
a=-b;
echo off
%set the sliders to change covariance and variance.
slide=uicontrol('Style','Slider','Position',[0 200 20 150],'CallBack','cla','String','Clear','Max',.75,'Min',-.75);
slide2=uicontrol('Style','Slider','Position',[0 20 20 150],'CallBack','cla','String','Clear','Max',1,'Min',-1);
%Read the slider values and call the interact_bases script.
set(slide,'CallBack','Interact_bayes')
set(slide2,'CallBack','Interact_bayes')
