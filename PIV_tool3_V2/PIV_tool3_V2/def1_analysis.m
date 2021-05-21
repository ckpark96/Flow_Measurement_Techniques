function [bufw,displx,disply,corrmap,sn]=def1_analysis(posx0,posy0,radx,rady,ima,imb);

d=5;
r=(d-1)/2;
wsx=2*radx+1;
wsy=2*rady+1;
displ=zeros(d,d,2);
posx=zeros(1,d); posy=zeros(1,d);
for cx=1:d
    for cy=1:d
        
        x0=posx0+(cx-r-1)*radx;
        y0=posy0+(cy-r-1)*rady;

        % perform correlation on deformed images
        [bufw,displx,disply,corrmap,sn]=std_analysis(x0,y0,radx,rady,ima,imb);
        displ(cy,cx,1)=displx;
        displ(cy,cx,2)=disply;
        posx(cx)=x0;
        posy(cy)=y0;
        
    end
end

% build dense predictor
[x,y]=meshgrid(posx,posy);
[X,Y]=meshgrid(posx0-radx:posx0+radx,posy0-rady:posy0+rady);
densepred(:,:,1)=interp2(x,y,displ(:,:,1),X,Y,'*linear');
densepred(:,:,2)=interp2(x,y,displ(:,:,2),X,Y,'*linear');


% deform images
[dimy,dimx]=size(X);
[tmpx,tmpy]=meshgrid(posx0-radx:posx0+radx,posy0-rady:posy0+rady);
xtransfA = tmpx + 0.5*densepred(:,:,1);
xtransfB = tmpx - 0.5*densepred(:,:,1);
ytransfA = tmpy + 0.5*densepred(:,:,2);
ytransfB = tmpy - 0.5*densepred(:,:,2);
clear tmpx tmpy

%wa=ima(posx0-r*radx:posx0+r*radx,posy0-r*rady:posy0+r*rady);
%wb=imb(posx0-r*radx:posx0+r*radx,posy0-r*rady:posy0+r*rady);

[dimypix,dimxpix]=size(ima);
[Xpix,Ypix]=meshgrid(1:dimxpix,1:dimypix);
%imdefa=sincinterp2(ima,xtransfA,ytransfA,ima,zeros(dimypix,dimxpix),3);
%imdefb=sincinterp2(imb,xtransfB,ytransfB,imb,zeros(dimypix,dimxpix),3);
wdefa=interp2(Xpix,Ypix,ima,xtransfA,ytransfA,'*linear',0);
wdefb=interp2(Xpix,Ypix,imb,xtransfB,ytransfB,'*linear',0);
%imb(find(isnan(imb)))=mean(ima);
%ima(find(isnan(ima)))=meangraya;

% perform correlation on deformed images
[bufw,displx,disply,corrmap,sn]=std_analysis2(posx0,posy0,radx,rady,wdefa,wdefb);

displx=displx+displ(r+1,r+1,1);
disply=disply+displ(r+1,r+1,2);


