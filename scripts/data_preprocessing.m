% % class_names={'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight',...
% %      'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', ...
% %      'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'};
% 
% 
% % 1 ‘Car’0,0,142;, 'truck'0,0,70;, 'bus'0,60,100;, 'train'0,80,100;, 'motorcycle'0,0,230;, 'bicycle'119,11,32;
% % 2. 'road'128,64,128;, 'sidewalk'244,35,232;, 'terrain'152,251,152;
% % 3. 'pole'153,153,153;, 'trafficlight'250,170,30;, 'trafficsign'220,220,0;, 'person'220,20,60;, ‘rider’255,0,0;
% % 4. 'building'70,70,70;, 'wall'102,102,156;, 'fence'190,153,153;
% % 5. Sky 70,130,180;
% % 6. 'vegetation'107,142,35;
% % 7. Void 0,0,0
% 
% clear all;close all;clc
% image_folder='/home/csdept/projects/pytorch-CycleGAN-and-pix2pix_sg2/datasets/dayton_pair/test';
% save_folder='/home/csdept/projects/pytorch-CycleGAN-and-pix2pix_sg2/datasets/dayton_pair_label/test';
% if ~isdir(save_folder)
%     mkdir(save_folder)
% end
% % load('/home/csdept/projects/iccv2019/segmentation_label.mat');
% Image =  dir( image_folder );  
% 
% parfor i = 1 : length( Image )
%     fprintf('%d / %d \n', i, length(Image));
%     if( isequal( Image( i ).name, '.' ) || isequal( Image( i ).name, '..' ))  
%         continue;
%     end
%     image_name=Image( i ).name;
%     if exist(fullfile(save_folder, image_name), 'file') == 0
%         image_path=fullfile(image_folder, image_name);
%         img=imread(image_path);
%     %     imshow(img)
%         image1=img(1:256,1:256,:);
%         image2=img(1:256,257:512,:);
%         image3=img(1:256,513:768,:);
%         image4=img(1:256,769:1024,:);
% 
%         for k=1:256
%             for j=1:256            
%                     if all(reshape(image4(k,j,:),[1,3])==[0,0,142]) || all(reshape(image4(k,j,:),[1,3])==[0,0,70]) || all(reshape(image4(k,j,:),[1,3])==[0,60,100]) || all(reshape(image4(k,j,:),[1,3])==[0,80,100])  || all(reshape(image4(k,j,:),[1,3])==[0,0,230]) || all(reshape(image4(k,j,:),[1,3])==[119,11,32])       
%                         image3(k,j,:) = [10,10,10];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[128,64,128]) || all(reshape(image4(k,j,:),[1,3])==[244,35,232]) || all(reshape(image4(k,j,:),[1,3])==[152,251,152])   
%                         image3(k,j,:) = [20,20,20];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[153,153,153]) || all(reshape(image4(k,j,:),[1,3])==[250,170,30]) || all(reshape(image4(k,j,:),[1,3])==[220,220,0]) || all(reshape(image4(k,j,:),[1,3])==[220,20,60]) || all(reshape(image4(k,j,:),[1,3])==[255,0,0])   
%                         image3(k,j,:) = [30,30,30];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[70,70,70]) || all(reshape(image4(k,j,:),[1,3])==[102,102,156]) || all(reshape(image4(k,j,:),[1,3])==[190,153,153])   
%                         image3(k,j,:) = [40,40,40];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[70,130,180])   
%                         image3(k,j,:) = [50,50,50];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[107,142,35])   
%                         image3(k,j,:) = [60,60,60];
% %                     elseif all(reshape(image4(k,j,:),[1,3])==[0,0,0])   
% %                         image3(k,j,:) = [70,70,70];
%                     end
% 
%             end
%         end
% 
%     im=[image1,image2,image3,image4];
% %     imshow(im)
%     imwrite(im, fullfile(save_folder, image_name));
%     end
% end

% class_names={'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight',...
%      'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', ...
%      'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'};


% 1 ‘Car’0,0,142;, 'truck'0,0,70;, 'bus'0,60,100;, 'train'0,80,100;, 'motorcycle'0,0,230;, 'bicycle'119,11,32;
% 2. 'road'128,64,128;, 'sidewalk'244,35,232;, 'terrain'152,251,152;
% 3. 'pole'153,153,153;, 'trafficlight'250,170,30;, 'trafficsign'220,220,0;, 'person'220,20,60;, ‘rider’255,0,0;
% 4. 'building'70,70,70;, 'wall'102,102,156;, 'fence'190,153,153;
% 5. Sky 70,130,180;
% 6. 'vegetation'107,142,35;
% 7. Void 0,0,0

clear all;close all;clc
% % image_folder='/home/csdept/projects/pytorch-CycleGAN-and-pix2pix_sg2/datasets/dayton_pair/train';
% % save_folder='/home/csdept/projects/pytorch-CycleGAN-and-pix2pix_sg2/datasets/dayton_pair_label_local_global/train';

image_folder='/data/Dataset_SVA_concatenate_test';
save_folder='/data/Hao_Tang/SVA_dataset_local_global/test';
if ~isdir(save_folder)
    mkdir(save_folder)
end
% load('/home/csdept/projects/iccv2019/segmentation_label.mat');
Image =  dir( image_folder );  

parfor i = 1 : length( Image )
    fprintf('%d / %d \n', i, length(Image));
    if( isequal( Image( i ).name, '.' ) || isequal( Image( i ).name, '..' ))  
        continue;
    end
    image_name=Image( i ).name;
    if exist(fullfile(save_folder, image_name), 'file') == 0
        image_path=fullfile(image_folder, image_name);
        img=imread(image_path);
    %     imshow(img)
        image1=img(1:256,1:256,:);
        image2=img(1:256,257:512,:);
        image3=img(1:256,513:768,:);
        image4=img(1:256,769:1024,:);
        image5=image3;
        image6=image4;

        for k=1:256
            for j=1:256            
                    if all(reshape(image4(k,j,:),[1,3])==[0,0,142]) || all(reshape(image4(k,j,:),[1,3])==[0,0,70]) || all(reshape(image4(k,j,:),[1,3])==[0,60,100]) || all(reshape(image4(k,j,:),[1,3])==[0,80,100])  || all(reshape(image4(k,j,:),[1,3])==[0,0,230]) || all(reshape(image4(k,j,:),[1,3])==[119,11,32])       
                        image6(k,j,:) = [10,10,10];
                    elseif all(reshape(image4(k,j,:),[1,3])==[128,64,128]) || all(reshape(image4(k,j,:),[1,3])==[244,35,232]) || all(reshape(image4(k,j,:),[1,3])==[152,251,152])   
                        image6(k,j,:) = [20,20,20];
                    elseif all(reshape(image4(k,j,:),[1,3])==[153,153,153]) || all(reshape(image4(k,j,:),[1,3])==[250,170,30]) || all(reshape(image4(k,j,:),[1,3])==[220,220,0]) || all(reshape(image4(k,j,:),[1,3])==[220,20,60]) || all(reshape(image4(k,j,:),[1,3])==[255,0,0])   
                        image6(k,j,:) = [30,30,30];
                    elseif all(reshape(image4(k,j,:),[1,3])==[70,70,70]) || all(reshape(image4(k,j,:),[1,3])==[102,102,156]) || all(reshape(image4(k,j,:),[1,3])==[190,153,153])   
                        image6(k,j,:) = [40,40,40];
                    elseif all(reshape(image4(k,j,:),[1,3])==[70,130,180])   
                        image6(k,j,:) = [50,50,50];
                    elseif all(reshape(image4(k,j,:),[1,3])==[107,142,35])   
                        image6(k,j,:) = [60,60,60];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[0,0,0])   
%                         image3(k,j,:) = [70,70,70];
                    end
                    
                    if all(reshape(image3(k,j,:),[1,3])==[0,0,142]) || all(reshape(image3(k,j,:),[1,3])==[0,0,70]) || all(reshape(image3(k,j,:),[1,3])==[0,60,100]) || all(reshape(image3(k,j,:),[1,3])==[0,80,100])  || all(reshape(image3(k,j,:),[1,3])==[0,0,230]) || all(reshape(image3(k,j,:),[1,3])==[119,11,32])       
                        image5(k,j,:) = [10,10,10];
                    elseif all(reshape(image3(k,j,:),[1,3])==[128,64,128]) || all(reshape(image3(k,j,:),[1,3])==[244,35,232]) || all(reshape(image3(k,j,:),[1,3])==[152,251,152])   
                        image5(k,j,:) = [20,20,20];
                    elseif all(reshape(image3(k,j,:),[1,3])==[153,153,153]) || all(reshape(image3(k,j,:),[1,3])==[250,170,30]) || all(reshape(image3(k,j,:),[1,3])==[220,220,0]) || all(reshape(image3(k,j,:),[1,3])==[220,20,60]) || all(reshape(image3(k,j,:),[1,3])==[255,0,0])   
                        image5(k,j,:) = [30,30,30];
                    elseif all(reshape(image3(k,j,:),[1,3])==[70,70,70]) || all(reshape(image3(k,j,:),[1,3])==[102,102,156]) || all(reshape(image3(k,j,:),[1,3])==[190,153,153])   
                        image5(k,j,:) = [40,40,40];
                    elseif all(reshape(image3(k,j,:),[1,3])==[70,130,180])   
                        image5(k,j,:) = [50,50,50];
                    elseif all(reshape(image3(k,j,:),[1,3])==[107,142,35])   
                        image5(k,j,:) = [60,60,60];
%                     elseif all(reshape(image4(k,j,:),[1,3])==[0,0,0])   
%                         image3(k,j,:) = [70,70,70];
                    end

            end
        end

    im=[image1,image2,image3,image4,image5,image6];
%      imshow(im)
    imwrite(im, fullfile(save_folder, image_name));
    end
end
