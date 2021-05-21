function [pt1,pt2] = fpg(fig)


%get original states
figure(fig)
root_unit=get(0,'Units');
fig_unit=get(fig,'Units');

set(fig,'Units','pixels');
set(0,'Units','pixels');
%scrn_pt = get(0,'PointerLocation')
[pt(1),pt(2)] = ginput(1);
loc = get(fig,'Position');
ax=get(fig,'CurrentAxes');
ax_unit=get(ax,'Units');
set(ax,'Units','pixels');
axloc = get(ax,'Position');
Ydir=get(ax,'YDir');
Xlim=get(ax,'XLim');
Ylim=get(ax,'Ylim');

% if strcmp('reverse',Ydir)==1;
% 	pt = [Xlim(1)+((Xlim(2)-Xlim(1)))*(scrn_pt(1) - loc(1) - axloc(1))/axloc(3), Ylim(2)-((Ylim(2)-Ylim(1)))*(scrn_pt(2) - loc(2) - axloc(2))/axloc(4)];
% else
%    pt = [Xlim(1)+((Xlim(2)-Xlim(1)))*(scrn_pt(1) - loc(1) - axloc(1))/axloc(3), Ylim(1)+((Ylim(2)-Ylim(1)))*(scrn_pt(2) - loc(2) - axloc(2))/axloc(4)];
% end;


if pt(1)>Xlim(1) & pt(1)<Xlim(2) & pt(2)>Ylim(1) & pt(2)<Ylim(2);
   pt1=round(pt(1));
   pt2=round(pt(2));
else
   pt1=[];
   pt2=[];
end;



%reset states
set(0,'Units',root_unit);
set(ax,'Units',ax_unit);
set(fig,'Units',fig_unit);