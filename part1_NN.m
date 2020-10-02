clc,
close all,
clear all,
%% PART I
for iimage = 1:50

subname = '.jpg';
iimage = num2str(iimage);
imagename = strcat(iimage,subname);
x = imread(imagename);
%figure, imshow(x)
imwrite(x,strcat('Input',iimage,'.jpg'));
%% Pre-Processing
x_gray = imresize(rgb2gray(x), [512 512]);
%figure, imshow(x_gray), impixelinfo
imwrite(x_gray,strcat('Gray',iimage,'.jpg'));
%% ROI
x_roi = x_gray(210:300,43:463);
%figure, imshow(x_roi), impixelinfo
imwrite(x_roi,strcat('ROI',iimage,'.jpg'));
%%
H = fspecial('motion',10,0);
MotionBlur = imfilter(x_roi,H,'replicate');
%figure, imshow(MotionBlur), title('MotionBlur'), impixelinfo
imwrite(MotionBlur,strcat('imfilter',iimage,'.jpg'));
%%
e = edge(MotionBlur,'canny');
%figure, imshow(e), title('edge'), impixelinfo
%% Threshold
[r c] = size(x_roi);

for i = 1:r
    for j = 1:c
        if MotionBlur(i,j) >= 40 && x_roi(i,j) <= 90
            
            MotionBlur(i,j) = 255;
            
        else
            
            MotionBlur(i,j) = 0;
            
        end
    end
end
%figure, imshow(MotionBlur), title('Threshold')
imwrite(MotionBlur,strcat('Threshold',iimage,'.jpg'));
%%
small_area = bwareaopen(MotionBlur,10);
%figure, imshow(small_area), title('small_area'), impixelinfo
imwrite(small_area,strcat('bwareaopen',iimage,'.jpg'));
%%
[Ilabel num] = bwlabel(small_area);
%%
Iprops = regionprops(Ilabel);
Ibox = [Iprops.BoundingBox];
Ibox = reshape(Ibox,[4 num]);
figure, imshow(x_roi)

hold on;
for cnt = 1:num
    rectangle('position',Ibox(:,cnt),'edgecolor','r');
end
%%
for iCentroid = 1:num
stats = regionprops(Ilabel, 'Centroid'); % หา Centroid
allCentroid = [stats.Centroid];

test = [stats(iCentroid,1).Centroid(1) stats(iCentroid,1).Centroid(2)];

datatraining = [21.28813559	6.338983051
21.16981132	88.0754717
35.47169811	16.8490566
35.54385965	27.05263158
35.5	98.95652174
50.17307692	47.21153846
50.07407407	88.42592593
64.44444444	36.74074074
64.53703704	78.07407407
79.04081633	57.24489796
79.1372549	68.60784314
115.9259259	36.96296296
130.3454545	27.23636364
130.3148148	78.42592593
144.8444444	68.73333333
145	17.02222222
145	47.56862745
145.0181818	88.43636364
159.5882353	6.980392157
159.4117647	57.45098039
159.5357143	99.03571429
210.7169811	36.94339623
210.8181818	88.69090909
225.0188679	47.81132075
239.5818182	27.87272727
239.5535714	99.19642857
253.9464286	57.875
254.0181818	68.90909091
268.1896552	7.379310345
268.1551724	17.62068966
268.4716981	78.79245283
305.375	27.92857143
305.4150943	99.37735849
319.9019608	7.666666667
319.9285714	57.85714286
319.82	88.86
334.1403509	37.49122807
334.1636364	69.03636364
348.5789474	48.01754386
348.4561404	78.94736842
362.7924528	17.90566038
399.6071429	48.21428571
399.962963	99.75925926
414.3333333	37.68888889
414.4375	89
428.4561404	17.96491228
443	7.924528302
442.8478261	69.23913043
457.220339	28.22033898
457.4313725	78.96078431];
%%
p = datatraining';
t = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
%%
[pn,ps] = mapminmax(p);
[tn,ts] = mapminmax(t);
%%
net=newff(pn,tn,[2 4 2],{'tansig' 'tansig' 'purelin'},'trainrp');

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
    neighborDistances(i,:) = sqrt(sortval(1:kn))
end

mini = min(neighborDistances);
out(iCentroid) = find(neighborDistances == mini);

%error1(iCentroid) = out(iCentroid) - T(iCentroid);

end

[rout cout] = size(out);
for iout = 0:cout-2
er(iout+1) = out(iout+2) - out(iout+1)
end
%% หาข้อซ้ำ ถ้า er มี -1 แสดงว่าซ้ำ
repaired = find(er == -1);
[rrepaired crepaired] = size(repaired);
countrepaired = 0;

for irepaired = 1:crepaired
    countrepaired = countrepaired + 1;
end
exceed = num - 50;
score = num - countrepaired - exceed;
%% หาข้อที่ไม่ได้ทำ
if exceed >= 1
    lost = 0;
else
    lost = 50 - num;
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
%fprintf('you get score from part I as %d per 50 score ',score)
end
