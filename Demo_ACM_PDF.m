% Meng xinchao
% 2024-01-20 PM



clc;clear all;close all;

Img=imread('F:\郑铁工作\个人工作\副教授\基于概率密度函数的活动轮廓模型\2.bmp');
Img = double(Img(:,:,1));

NumIter = 1000; %iterations
timestep=0.1; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 3.5;%size of kernel
epsilon = 1;
c0 = 2; % the constant value 
lambda1=1.0;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight


figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
[Height Wide] = size(Img);
[xx yy] = meshgrid(1:Wide,1:Height);
phi = (sqrt(((xx -32).^2 + (yy - 40).^2 )) - 30);
phi = sign(phi).*c0;


Ksigma=fspecial('gaussian',round(2*sigma)*2 + 1,sigma); %  kernel
ONE=ones(size(Img));
KONE = imfilter(ONE,Ksigma,'replicate');  
KI = imfilter(Img,Ksigma,'replicate');  
KI2 = imfilter(Img.^2,Ksigma,'replicate'); 



figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,
hold on,[c,h] = contour(phi,[0 0],'g','linewidth',1.5); hold off

pause(0.5)

tic
for iter = 1:NumIter
    phi =evolution_ACM_PDF(Img,phi,epsilon,Ksigma,KONE,KI,KI2,mu,nu,lambda1,lambda2,timestep,alf);

    if(mod(iter,10) == 0)
        figure(2),
%         set(gca,'Position',get(0,'ScreenSize'))
%         set(gca, 'Position', [0,0,1,1]);
        imagesc(uint8(Img),[0 255]),colormap(gray),hold on,axis off;axis equal,title(num2str(iter))
        [c,h] = contour(phi,[0 0],'r','linewidth',1.5); hold off
%         set(gca,'LooseInset', get(gca,'TightInset'))
%         set(gca, 'Position', get(0, 'Screensize'));
%         set(gca, 'LooseInset', [0,0,0,0])

%         pause(0.02);
    end

end
toc

figure,imshow(Img,[ ]);colormap(gray(1));hold on;
[c,h] =contour(phi,[1 1],'w','linewidth',1);

seg = (phi<0);
imshow(seg);

figure;
mesh(phi);


