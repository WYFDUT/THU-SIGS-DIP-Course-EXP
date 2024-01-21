I=imread('foggy1.jpg');
I=im2double(I);
tmap=imread('foggy1t.png');
tmap=im2double(tmap);
tic
tmap_ref = softmatting( I,tmap );
toc
imwrite(tmap_ref,'foggy1SM.jpg')
