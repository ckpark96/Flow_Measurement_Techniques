%[x_mp,y_mp]=PIVT_MousePush(HandlePIVT_Main);

% 1-step deformed cross correlation
%[bufw2,displx2,disply2,corrmap2,sn2]=def1_analysis(x0,y0,radx,rady,ima,imb);
   
figure(HandlePIVT_Main), hold on, 
% x0=x_mp;
% y0=y_mp;
sc=10;
labcol{1}='g';
labcol{2}='y';
labcol{3}='r';
if sn2>2
    h1=quiver(x0,y0,sc*displx2,sc*disply2,'color',labcol{1},'linewidth',2);
    adjust_quiver_arrowhead_size(h1, 2.5);
%    line([x0 x0+sc*displx2],[y0 y0+sc*disply2],'color',labcol{1})
%    plot(x0,y0,[ labcol{1} '.'])
elseif (sn2<=2 && sn2>1.5)
    h1=quiver(x0,y0,sc*displx2,sc*disply2,'color',labcol{2},'linewidth',2);
    adjust_quiver_arrowhead_size(h1, 2.5);
%    line([x0 x0+sc*displx2],[y0 y0+sc*disply2],'color',labcol{2})
%    plot(x0,y0,[ labcol{2} '.'])
else
    h1=quiver(x0,y0,sc*displx2,sc*disply2,'color',labcol{3},'linewidth',2);
    adjust_quiver_arrowhead_size(h1, 2.5);
%    line([x0 x0+sc*displx2],[y0 y0+sc*disply2],'color',labcol{3})
%    plot(x0,y0,[ labcol{3} '.'])
end

hold off

set(HandlePIVT_DisplValueX,'String',num2str(displx2));
set(HandlePIVT_DisplValueY,'String',num2str(disply2));
