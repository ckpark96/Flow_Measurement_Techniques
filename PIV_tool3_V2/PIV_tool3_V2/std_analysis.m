function [bufw,displx,disply,corrmap,sn]=std_analysis(x0,y0,radx,rady,ima,imb);

wsx=radx*2+1;
wsy=rady*2+1;

wa=ima(y0-rady:y0+rady,x0-radx:x0+radx);
wb=imb(y0-rady:y0+rady,x0-radx:x0+radx);

a2=wa-mean(wa(:));
b2=wb-mean(wb(:));

%For fast FFT Nfft needs to be close to 2^N but smaller, e.g. wsx=15,31,63
b2 = b2(end:-1:1,end:-1:1);
Nfftx=2^( ceil(log2(wsx)) );
Nffty=2^( ceil(log2(wsy)) );

% FFT of both:
ffta=fft2(a2,Nffty,Nfftx);
fftb=fft2(b2,Nffty,Nfftx);

% Real part of an Inverse FFT of a conjugate multiplication and normalize:
c = real(ifft2(ffta.*fftb));
a2=a2(:); b2=b2(:); s1=sum(a2.^2); s2=sum(b2.^2);
c=c/sqrt(s1*s2);
% c = real(ifft2(ffta.*fftb))/(wsx*wsy);
corrmap2=circshift(c,[-rady -radx]);
corrmap=corrmap2(1:wsy,1:wsx);
% corrmap=corrmap2(round(end/2)-radx:round(end/2)+radx,round(end/2)-rady:round(end/2)+rady);
% size(corrrmap2)
% size(corrmap)

clear bufw;
dum=flipud(wb);
bufw(:,:,1)=flipud(wa);
bufw(:,:,2)=dum;
% bufw(:,:,3)=flipud(wb);
bufw(:,:,3)=dum;
bufw=bufw/(max(bufw(:)));


corrmap(find(isnan(corrmap)))=0;
[peaks,sn]=analyse_map_tool(corrmap,1,3,1);
sn=max(sn,1);

[displx,disply]=gaussian(peaks,corrmap);
% corrmap((isnan(corrmap)))=0;
% MapRes=AnalyzeMap2(corrmap);
% displx=MapRes(1);
% disply=MapRes(2);
% sn=MapRes(3)/MapRes(6);
