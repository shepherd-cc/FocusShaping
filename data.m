%% 
%z=0
clear
clc
NA = 0.95;
theta = asin(NA);
lamda=532e-9;
k=2*pi/lamda;
r = 0:1e-6/100:1e-6;
ringnum = 25;
T = ones(1,ringnum);
z = 0*lamda;
%radial polarized
for i = 1:size(T,2)
Ez(:,i)=integral(@(a)(exp(-sin(a).^2*0/(sin(theta))^2).*T(i)*besselj(0,k*r.*sin(a)).*(sin(a).^2).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
Er(:,i)=integral(@(a)(exp(-sin(a).^2*0/(sin(theta))^2).*T(i)*besselj(1,k*r.*sin(a)).*0.5.*sin(2*a).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
end

plot(1:101,abs(sum(Ez,2)).^2,1:101,abs(sum(Er,2)).^2,1:101,abs(sum(Ez,2)).^2+abs(sum(Er,2)).^2)

point = [1,12];
%point(3:103-2*point(size(point,2)))=(2*point(size(point,2))+1:1:101);
Y = [1,0.5];
for i = 1:size(T,2)%
    for j = 1:size(point,2)
        a(j,i)=Ez(point(j),i);
        b(j,i)=Er(point(j),i);
    end  
    %b(1,i)=Er(point(1),i);
    %b(2,i)=Er(point(2),i);  
end

can1 = single(a);
can2 = single(b);
X = single(1);
Y = single(Y);

Ez=sum(Ez,2);
Er=sum(Er,2);
I = abs(Ez).^2+abs(Er).^2;
nor = I(1);
save(['D:/FocusShaping/','optimization.mat'], 'X','Y','can1','can2','nor')


T=T(size(T,1),:);
T = T/max(abs(T));
plot(T)



%%

%r=0
clear
clc
NA = 0.95;
theta = asin(NA);
lamda=532e-9;
k=2*pi/lamda;
z = -10*lamda:10*lamda/100:10*lamda;
ringnum = 25;
T = ones(1,ringnum);

r = 0;
for i = 1:size(T,2)
Ez1(:,i)=integral(@(a)(T(i)*besselj(0,k*r.*sin(a)).*(sin(a).^2).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
Er1(:,i)=integral(@(a)(T(i)*besselj(1,k*r.*sin(a)).*0.5.*sin(2*a).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
end

r = 1e-6/100*11;
for i = 1:size(T,2)
Ez2(:,i)=integral(@(a)(T(i)*besselj(0,k*r.*sin(a)).*(sin(a).^2).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
Er2(:,i)=integral(@(a)(T(i)*besselj(1,k*r.*sin(a)).*0.5.*sin(2*a).*sqrt(cos(a)).*exp(1i*k*z.*cos(a))),(i-1)*theta/size(T,2),i*theta/size(T,2),'ArrayValued',true);
end

point1 = [101,111,121,131,141,151,161,171,181,191,201];
point2 = [101,141,181];
Y = [1,0.5,0,0.5,1,0.5,0,0.5,1,0.5,0];
Y(size(Y,2)+1:size(Y,2)+size(point2,2)) = 0.5;

for i = 1:size(T,2)%
    for j = 1:size(point1,2)
        a(j,i)=(Ez1(point1(j),i));
    end
    for j = 1:size(point2,2)
        b(j,i)=(Er2(point2(j),i));
        c(j,i)=(Ez2(point2(j),i));
    end
end
can1 = single(a);
can2 = single(b);
can3 = single(c);
X = single(1);
Y = single(Y);

Ez1=sum(Ez1,2);
Er1=sum(Er1,2);
Ez2=sum(Ez2,2);
Er2=sum(Er2,2);
plot((abs(Ez1)).^2+(abs(Er1)).^2)
plot((abs(Ez2)).^2+(abs(Er2)).^2)

save(['D:/FocusShaping/','optimization.mat'], 'X','Y','can1','can2','can3')


T=T(size(T,1),:);
T = T/max(abs(T));
plot(T)
plot(maxvalue)
