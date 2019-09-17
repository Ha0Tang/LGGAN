%     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
%     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
%     
%     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
%     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
%     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
%     
%     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
%     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
%     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
% 
%     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
%     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
% 
%     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
% 
%     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
%     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
%     
%     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
%     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
%     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
%     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
%     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
%     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),

% class information can be found https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

clear all;close all;clc
% image_folder='/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes/leftImg8bit/val';
% segmeantaion_folder = '/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes/gtFine/val';
% save_folder='/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes_for_training/val';

image_folder='/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes/leftImg8bit/train';
segmeantaion_folder = '/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes/gtFine/train';
save_folder='/home/htang/projects/LocalGlobalGAN/datasets/Cityscapes_for_training/train';
if ~isfolder(save_folder)
    mkdir(save_folder)
end

Image =  dir( image_folder );  

parfor i = 1 : length( Image )
% for i = 1 : length( Image )
    fprintf('%d / %d \n', i, length(Image));
    if( isequal( Image( i ).name, '.' ) || isequal( Image( i ).name, '..' ))  
        continue;
    end
    image_name=Image( i ).name;
    image_path=fullfile(image_folder, image_name);
    segmeantion_path = fullfile(segmeantaion_folder, strcat(image_name(1:length(image_name)-length('leftImg8bit.png')), 'gtFine_color.png'));
        img=imread(image_path);
        seg=imread(segmeantion_path);
%          imshow(img)
%          imshow(seg)
        img=imresize(img,[256,512], 'nearest');
        seg=imresize(seg,[256,512], 'nearest'); % color seg        
        label=seg;

        for k=1:256
            for j=1:512            
                if all(reshape(seg(k,j,:),[1,3])==[128, 64,128]) || all(reshape(seg(k,j,:),[1,3])==[244, 35,232])     
                    label(k,j,:) = [10,10,10];
                elseif all(reshape(seg(k,j,:),[1,3])==[70, 70, 70]) || all(reshape(seg(k,j,:),[1,3])==[102,102,156]) || all(reshape(seg(k,j,:),[1,3])==[190,153,153])   
                    label(k,j,:) = [20,20,20];
                elseif all(reshape(seg(k,j,:),[1,3])==[153,153,153]) || all(reshape(seg(k,j,:),[1,3])==[250,170, 30]) || all(reshape(seg(k,j,:),[1,3])==[220,220,  0])  
                    label(k,j,:) = [30,30,30];
                elseif all(reshape(seg(k,j,:),[1,3])==[107,142, 35]) || all(reshape(seg(k,j,:),[1,3])==[152,251,152])  
                    label(k,j,:) = [40,40,40];
                elseif all(reshape(seg(k,j,:),[1,3])==[70,130,180])   
                    label(k,j,:) = [50,50,50];
                elseif all(reshape(seg(k,j,:),[1,3])==[220, 20, 60]) || all(reshape(seg(k,j,:),[1,3])==[255,  0,  0])   
                    label(k,j,:) = [60,60,60];
                elseif all(reshape(seg(k,j,:),[1,3])==[0,  0,142]) || all(reshape(seg(k,j,:),[1,3])==[ 0,  0, 70]) || all(reshape(seg(k,j,:),[1,3])==[0, 60,100]) || all(reshape(seg(k,j,:),[1,3])==[0, 80,100]) || all(reshape(seg(k,j,:),[1,3])==[0,  0,230]) || all(reshape(seg(k,j,:),[1,3])==[119, 11, 32])   
                    label(k,j,:) = [70,70,70];
                else 
                    label(k,j,:) = [80,80,80];
                end
                
            end
        end

    im=[seg,img,label];
%      imshow(im)
    imwrite(im, fullfile(save_folder, image_name));
end
