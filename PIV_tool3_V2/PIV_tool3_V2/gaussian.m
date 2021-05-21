function [displx,disply]=gaussian(peaks,corrmap)

posx=peaks(1,1);
posy=peaks(1,2);
R(1)=corrmap(posy,posx);				%Rcenter
R(2)=corrmap(posy,posx-1);	%Rleft
R(3)=corrmap(posy,posx+1);	%Rright
R(4)=corrmap(posy-1,posx);	%Rtop
R(5)=corrmap(posy+1,posx);	%Rbottom
shiftval=0;
if min(R)<=0
   shiftval=-min(R)+0.00001;
end
R=R+shiftval;

if ( R(1)>R(2) & R(1)>R(3) & R(1)>R(4) & R(1)>R(5) )
   
   [wsy,wsx]=size(corrmap);
   rx=floor((wsx-1)/2);
   ry=floor((wsy-1)/2);
   %displx=posx-rx-1+.50*(log(R(2))-log(R(3)))/(log(R(2))-2*log(R(1))+log(R(3)));
   %disply=posy-ry-1+.50*(log(R(5))-log(R(4)))/(log(R(5))-2*log(R(1))+log(R(4)));
   displx=posx-rx-1+.50*(log(R(2))-log(R(3)))/(log(R(2))-2*log(R(1))+log(R(3)));
   disply=posy-ry-1+.50*(log(R(4))-log(R(5)))/(log(R(4))-2*log(R(1))+log(R(5)));
else
   displx=0;
   disply=0;
end;


%just for checking!
%posx=posx+.50*(log(R(2))-log(R(3)))/(log(R(2))-2*log(R(1))+log(R(3)))
%posy=posy+.50*(log(R(4))-log(R(5)))/(log(R(4))-2*log(R(1))+log(R(5)))

