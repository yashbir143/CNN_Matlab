%% PART III
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
imwrite(x_gray,strcat('GrayIII',iimage,'.jpg'));

x_roi_III = x_gray(336:473,300:450);
%figure, imshow(x_roi_III), title('ROI II'), impixelinfo
imwrite(x_roi_III,strcat('roi_III',iimage,'.jpg'));
%%
H = fspecial('motion',10,0);
MotionBlurIII = imfilter(x_roi_III,H,'replicate');
%figure, imshow(MotionBlurIII), title('MotionBlurII'), impixelinfo
imwrite(MotionBlurIII,strcat('MotionBlurIII',iimage,'.jpg'));
%% Threshold
[rMotionBlurIII cMotionBlurIII] = size(MotionBlurIII);

for iMotionBlurIII = 1:rMotionBlurIII
    for jMotionBlurIII = 1:cMotionBlurIII
        if MotionBlurIII(iMotionBlurIII,jMotionBlurIII) >= 40 && MotionBlurIII(iMotionBlurIII,jMotionBlurIII) <= 99
            
            MotionBlurIII(iMotionBlurIII,jMotionBlurIII) = 255;
            
        else
            
            MotionBlurIII(iMotionBlurIII,jMotionBlurIII) = 0;
            
        end
    end
end
%figure, imshow(MotionBlurIII), title('Threshold II')
imwrite(MotionBlurIII,strcat('Threshold_III',iimage,'.jpg'));
%%
small_area_III = bwareaopen(MotionBlurIII,0);
%figure, imshow(small_area_III), title('small_area_II'), impixelinfo
imwrite(small_area_III,strcat('small_area_III',iimage,'.jpg'));
%%
[Ilabel_III num_III] = bwlabel(small_area_III);
%%
Iprops_III = regionprops(Ilabel_III);
Ibox_III = [Iprops_III.BoundingBox];
Ibox_III = reshape(Ibox_III,[4 num_III]);
figure, imshow(x_roi_III)

hold on;
for cnt = 1:num_III
    rectangle('position',Ibox_III(:,cnt),'edgecolor','r');
end
%%
for iCentroid = 1:num_III
stats = regionprops(Ilabel_III, 'Centroid'); % หา Centroid
allCentroid = [stats.Centroid];

test = [stats(iCentroid,1).Centroid(1) stats(iCentroid,1).Centroid(2)];

datatraining = [40.42857143	39.92857143
40.6	51.52727273
54.21666667	16.68333333
54.81034483	5.551724138
55.03571429	143.8392857
55.30508475	155.1694915
68.74576271	62.79661017
68.7962963	120.3703704
69.19672131	28.75409836
83.42372881	132.0508475
83.53333333	85.96666667
97.56603774	74.58490566
97.48275862	97.82758621
97.28571429	108.9107143
151.1730769	5.519230769
150.9666667	16.85
151.4423077	155.2692308
164.6140351	28.8245614
164.6666667	40.10526316
164.4590164	120.8032787
178.9285714	51.91071429
179.537037	62.90740741
179.6949153	143.9830508
194.2333333	74.85
193.9322034	86.01694915
194.3275862	132.0344828
207.3571429	97.89285714
207.5283019	109.0377358];
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
out(iCentroid) = find(neighborDistances == mini);


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
exceed = (num_III/2) - 14;
score = (num_III/2) - countrepaired - exceed;
%% หาข้อที่ไม่ได้ทำ
if exceed >= 1
    lost = 0;
else
    lost = 14 - (num_III/2);
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
%fprintf('you get score from part III as %d per 14 score ',score);

end

