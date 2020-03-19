clear
clc
dataset='test';     
ori_path=fullfile('./ori_data',dataset);
s_path=fullfile('./data',dataset);
file=dir(fullfile(ori_path,'*.mat'));
scale=32;
for s=1:length(file)
    name=file(s).name;
    display(name);
    load(fullfile(ori_path,name))
    if strfind(name,'HCI')==1
        data=lf_data(:,:,:,:,:);
    else
        data=LF(:,:,:,:,:); 
    end
    c_data =data(2:8,2:8,:,:,:);
    [LF_hr,LF_lr,~]=lf_downsample_bicubic(c_data,scale);            
    save_path=fullfile(s_path,['x',num2str(scale)],name);
    save(save_path,'LF_hr','LF_lr')
end
