% INPUT:
% I is the input image of double type, eg. I=im2double(imread('foggy1.jpg'))
% tmap is the raw transmission map
% OUTPUT:
% tmap_ref is the refined transmission map
function tmap_ref = softmatting( I,tmap )

% this function is used to refine the raw transmission map in Image dehazing.

  if (~exist('epsilon','var'))
    epsilon=0.0000001;
  end  
  if (isempty(epsilon))
    epsilon=0.0000001;
  end
  if (~exist('win_size','var'))
    win_size=1;
  end     
  if (isempty(win_size))
    win_size=1;
  end     
  if (~exist('lambda','var'))
    lambda=0.001;
  end  
  if (isempty(lambda))
    lambda=0.001;
  end
  
  [h,w,~]=size(I);
  img_size=w*h;
  
  win_b = zeros(img_size,1);

for ci=1:h
    for cj=1:w
        if(rem(ci-8,15)<1)
            if(rem(cj-8,15)<1)
                win_b(ci*w+cj)=tmap(ci*w+cj);
            end
        end
       
    end
end
  %======================get laplace L============================
  A=getLaplacian(I,epsilon,win_size);
  %====================solve refined transmission map====================  
  D=spdiags(win_b(:),0,img_size,img_size);
  x=(A+lambda*D)\(lambda*win_b(:).*win_b(:));
  tmap_ref=max(min(reshape(x,h,w),1),0);
  %=============================================================

end
