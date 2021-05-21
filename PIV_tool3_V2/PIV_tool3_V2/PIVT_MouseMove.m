
[x0,y0]=fpg(HandlePIVT_Main);
set(HandlePIVT_CoordPointY,'String',num2str(y0));
set(HandlePIVT_CoordPointX,'String',num2str(x0));
%set(HandlePIVT_CoordValue,'String',num2str(ima_b(y0,x0)));

wsx=eval(get(HandlePIVT_WSX,'String'));
wsy=eval(get(HandlePIVT_WSY,'String'));
radx=round((wsx-1)/2);
rady=round((wsy-1)/2);

a2=wa;
b2=wb;

% Rotate the second image ( = conjugate FFT)
b2 = b2(end:-1:1,end:-1:1);


%For fast FFT Nfft needs to be close to 2^N but smaller, e.g. wsx=15,31,63
Nfftx=2^( ceil(log2(wsx)) );
Nffty=2^( ceil(log2(wsy)) );

% FFT of both:
ffta=fft2(a2,Nffty,Nfftx);
fftb=fft2(b2,Nffty,Nfftx);

% Real part of an Inverse FFT of a conjugate multiplication and normalize:
c = real(ifft2(ffta.*fftb))/(wsx*wsy);
% corrmap2=circshift(c,[-radx -rady]);
corrmap=c(1:wsy,1:wsx);

% figure(33); clf
% imagesc(c); colorbar
% pause
% corrmap=c(round(wsy/2)-rady:round(wsy/2)+rady,round(wsx/2)-radx:round(wsx/2)+radx);

corrmap(find(isnan(corrmap)))=0;
[peaks,sn]=analyse_map_tool(corrmap,1,3,1);
sn=max(sn,1);


figure(HandlePIVT_CorrMap), subplot(1,2,1), imagesc(bufw), axis equal, title(['wa+wb']), drawnow;
if CorrmapFlag==0
    figure(HandlePIVT_CorrMap), subplot(1,2,2), imagesc(flipud(corrmap)), axis equal, title(['Cmap S/N=' num2str(0.1*round(sn*10))]), drawnow;
end;
if CorrmapFlag==1
    figure(HandlePIVT_CorrMap), subplot(1,2,2), surf(flipud(corrmap)), title(['Cmap S/N=' num2str(0.1*round(sn*10))]), drawnow;
end;

