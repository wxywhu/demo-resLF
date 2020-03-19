function [LF_hr,LF_lr,LF_bic] = lf_downsample_bicubic(lf_data,scale)

[U,V,H,W,C] = size(lf_data);
LF_hr = zeros([U,V,(H-mod(H,scale)),(W-mod(W,scale)),C],'uint8');
LF_lr = zeros([U,V,(H-mod(H,scale))/scale,(W-mod(W,scale))/scale,C],'uint8');
LF_bic=LF_hr;
for u = 1:U
    for v = 1:V
        sub_img = squeeze(lf_data(u,v,:,:,:));
        sub_img = modcrop(sub_img,scale);
        LF_hr(u,v,:,:,:)=sub_img;
        LF_lr(u,v,:,:,:)=imresize(sub_img,1/scale,'bicubic');
        LF_bic(u,v,:,:,:)=imresize(imresize(sub_img,1/scale,'bicubic'),scale,'bicubic');
    end
end
end