syms x1 y1 theta1 x2 y2 theta2 z11 z12 zx zy 

X1=[cos(theta1), -sin(theta1), x1;
    sin(theta1), cos(theta1), y1;
    0,0,1]

X2=[cos(theta2), -sin(theta2), x2;
    sin(theta2), cos(theta2), y2;
    0,0,1]

Z=[z11 z12 zx;-z12 z11 zy; 0 0 1]
Z=Z^(-1)

X1_inv=simplify(X1^-1)

Z_bar=simplify(Z*X1_inv*X2)


x3=simplify(Z_bar(1,3))
y3=simplify(Z_bar(2,3))
theta3=simplify(atan(Z_bar(2,1)/ Z_bar(1,1)))

J1=simplify([diff(x3,x1) diff(x3,y1) diff(x3,theta1) ;
   diff(y3,x1) diff(y3,y1) diff(y3,theta1) ;
   diff(theta3,x1) diff(theta3,y1) diff(theta3,theta1) ;])

J2=simplify([ diff(x3,x2) diff(x3,y2) diff(x3,theta2);
    diff(y3,x2) diff(y3,y2) diff(y3,theta2);
    diff(theta3,x2) diff(theta3,y2) diff(theta3,theta2);])

x1=1;
y1=3;
theta1=0.0543;

x2=4;
y2=1;
theta2=0.26;

theta=0.26;
zx=3;
zy=-2;

z11=cos(theta)
z12=-sin(theta)
 %%
double(subs(J1))

syms a b c d x y 
A=[b+1 -b;-b b]

A^(-1)*[x;-x]