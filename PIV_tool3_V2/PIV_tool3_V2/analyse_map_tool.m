function [peaks,sn]=analyse_map_tool(corrmap,iter,npeaks,winiterconst)


[wsy,wsx]=size(corrmap);
Cmean=mean(corrmap(:));
rx=(wsx)/2;
ry=(wsy)/2;
srx=round(rx/2);
sry=round(rx/2);
invsignx=zeros(wsy,wsx);											invsigny=zeros(wsy,wsx);
invsignx(:,2:wsx-1)=diff(sign(diff(corrmap,1,2)),1,2);	invsigny(2:wsy-1,:)=diff(sign(diff(corrmap,1,1)),1,1);
bufx=-0.5*invsignx;													bufy=-0.5*invsigny;
bufx=floor(bufx+abs(bufx));										bufy=floor(bufy+abs(bufy));

peaks=bufx.*bufy/4;
peaksval=peaks.*corrmap;
[J,I,V]=find(peaksval);

numpeaks=length(I);

if numpeaks==0
   peak1=[rx+1,ry+1,0];
   peak2=[rx+1,ry+1,1];
   peaks=peak1;
%disp('no peaks')   
end;

if numpeaks==1
   peak1=[I,J,V];
   peak2=[I,J,V];
   peaks=peak1;
%disp('one peak')   
end;

if numpeaks>1
   peakslist=flipud(sortrows([I,J,V],3));
   peaks=peakslist(1:min(npeaks,numpeaks),:);
   peak1=peaks(1,:);
   peak2=peaks(2,:);
end;
sn=peak1(:,3)/peak2(:,3);