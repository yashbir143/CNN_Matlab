%% PART II
%% ROI
clc,
close all,
clear all,
%%
for iimage = 1:50

subname = '.jpg';
iimage = num2str(iimage);
imagename = strcat(iimage,subname);
x = imread(imagename);
%figure, imshow(x)
%% Pre-Processing
x_gray = imresize(rgb2gray(x), [512 512]);
%figure, imshow(x_gray), impixelinfo
imwrite(x_gray,strcat('grayII',iimage,'.jpg'));
%%
x_roi_II = x_gray(338:473,45:245);
%figure, imshow(x_roi_II), title('ROI II'), impixelinfo
imwrite(x_roi_II,strcat('ROI_II',iimage,'.jpg'));
%%
H = fspecial('motion',10,0);
MotionBlurII = imfilter(x_roi_II,H,'replicate');
%figure, imshow(MotionBlurII), title('MotionBlurII'), impixelinfo
imwrite(MotionBlurII,strcat('imfilter',iimage,'.jpg'));
%% Threshold
[rMotionBlurII cMotionBlurII] = size(MotionBlurII);

for iMotionBlurII = 1:rMotionBlurII
    for jMotionBlurII = 1:cMotionBlurII
        if MotionBlurII(iMotionBlurII,jMotionBlurII) >= 30 && MotionBlurII(iMotionBlurII,jMotionBlurII) <= 125
            
            MotionBlurII(iMotionBlurII,jMotionBlurII) = 255;
            
        else
            
            MotionBlurII(iMotionBlurII,jMotionBlurII) = 0;
            
        end
    end
end
%figure, imshow(MotionBlurII), title('Threshold II')
imwrite(MotionBlurII,strcat('Threshold',iimage,'.jpg'));
%%
small_area_II = bwareaopen(MotionBlurII,20);
%figure, imshow(small_area_II), title('small_area_II'), impixelinfo
imwrite(small_area_II,strcat('small_area_II',iimage,'.jpg'));
%%
[Ilabel_II num_II] = bwlabel(small_area_II);
%%
Iprops_II = regionprops(Ilabel_II);
Ibox_II = [Iprops_II.BoundingBox];
Ibox_II = reshape(Ibox_II,[4 num_II]);
figure, imshow(x_roi_II)

hold on;
for cnt = 1:num_II
    rectangle('position',Ibox_II(:,cnt),'edgecolor','r');
end
%%
for iCentroid = 1:num_II
stats = regionprops(Ilabel_II, 'Centroid'); % หา Centroid
allCentroid = [stats.Centroid];

test = [stats(iCentroid,1).Centroid(1) stats(iCentroid,1).Centroid(2)];

datatraining = [29.70408163	47.02040816
30.41573034	6.168539326
30.18085106	67.88297872
30.18478261	88.11956522
30	108.7948718
30.3258427	129.0561798
30.70930233	16.3255814
30.48863636	119.0909091
30.97333333	139.52
30.72941176	150.0588235
58.64646465	26.81818182
58.62886598	57.07216495
59.8372093	36.69767442
59.45555556	77.42222222
59.31707317	98.32926829
105.311828	27.05376344
105.3297872	47.08510638
104.875	98.5125
105.3378378	77.81081081
105.5714286	119.1098901
105.7466667	139.52
105.5176471	149.9647059
133.9183673	6.030612245
134.2291667	16.66666667
134.0309278	36.82474227
134.2045455	57.26136364
134.375	67.73863636
133.9578947	87.97894737
133.4831461	108.7303371
134.2413793	128.7931034
180.2857143	6.21978022
180.3444444	68.07777778
180.3736264	129.0879121
180.5444444	27.13333333
180.6086957	47.10869565
180.4942529	88.1954023
180.6304348	149.9347826
208.9894737	16.68421053
208.6831683	36.84158416
209.0722892	57.34939759
209.021978	77.73626374
209.5	98.41860465
209.1190476	139.3095238
209.25	108.6052632
209.3529412	118.8235294];
%%
p = datatraining';
t = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
    

[pn,ps] = mapminmax(p);
[tn,ts] = mapminmax(t);
%%
net=newff(pn,tn,[4 5 1],{'tansig' 'tansig' 'purelin'},'trainrp');

%%
net.trainParam.show = 10;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
%% training
%net = newp(p,t);
net=train(net,pn,tn);

%% simulation
y1=sim(net,test);
y = mapminmax('reverse',y1,ts)
%%
dataMatrix = test;
queryMatrix = datatraining;
kn = 1;

neighborIds = zeros(size(queryMatrix,1),kn);
neighborDistances = neighborIds;

numDataVectors = size(dataMatrix,1);
numQueryVectors = size(queryMatrix,1);

for i=1:numQueryVectors,
    dist = sum((repmat(queryMatrix(i,:),numDataVectors,1)-dataMatrix).^2,2);
    [sortval sortpos] = sort(dist,'ascend');
     neighborIds(i,:) = sortpos(1:kn);
    neighborDistances(i,:) = sqrt(sortval(1:kn));
end

mini = min(neighborDistances);
out(iCentroid) = find(neighborDistances == mini)


end

[rout cout] = size(out);
for iout = 0:cout-2
er(iout+1) = out(iout+2) - out(iout+1);
end
%% หาข้อซ้ำ ถ้า er มี -1 แสดงว่าซ้ำ
repaired = find(er == -1);
[rrepaired crepaired] = size(repaired);
countrepaired = 0;

for irepaired = 1:crepaired
    countrepaired = countrepaired + 1;
end
exceed = num_II - 45;
score = num_II - countrepaired - exceed;
%% หาข้อที่ไม่ได้ทำ
if exceed >= 1
    lost = 0;
else
    lost = 45 - num_II;
end
score = score - lost;
%% หาข้อที่ผิด
[rer cer] = size(er);
count_error = 0;
for ier = 1:cer
if er(ier) >= 2
    count_error = count_error + 1;
end
end
score = score - lost - count_error;
%fprintf('you get score from part II as %d per 45 score ',score)
end