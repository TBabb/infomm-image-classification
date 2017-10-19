function s = imag_class(str,v,varargin)
% IMAG_CLASS tests supervised classification on the Yale face database.
%
% S = IMAG_CLASS('numofpeople',N) takes N random people from the 38 people 
% in the Yale database and uses 10 random images from each person to build the 
% test set.
% It outputs the percentage success rate, S, of the algorithm.
%
% S = IMAG_CLASS('people',V) takes 10 random images from each person 
% labelled in V to build the test set.
% It outputs the percentage success rate, S, of the algorithm.
% 
% Optional Arguments:
% - (...,'numofcoeffs',N) specifies the number, N, of coefficients that PCA
% uses. By default this is 100.
%
% - (...,'numofneighs',N) specifies the number, N, of neighbours that the
% k-nearest neighbours algorithm selects. By default this is 3.

%% Inputs
load YaleB_32x32.mat fea gnd;

switch lowercase(str)
    case 'people'
        ppl = v;
    case 'numofpeople'
        num = v;
        ppl = randperm(38,num);
    otherwise
        ppl = randperm(38,10);
end

if nargin == 2
    NumCoeffs = 100;
    NumNeighbours = 3;
elseif nargin == 4 &&
    switch lowercase(varargin{1})
        case 'numofcoeffs'
            NumCoeffs = varargin{2};
            NumNeighbours = 3;
        case 'numofneighs'
            NumNeighbours = varargin{2};
            NumCoeffs = 100;
    end
elseif nargin == 6
    switch lowercase(varargin{1})
        case 'numofcoeffs'
            NumCoeffs = varargin{2};
            NumNeighbours = varargin{4};
        case 'numofneighs'
            NumNeighbours = varargin{2};
            NumCoeffs = varargin{4};
    end
end

%% Selection of the Training and the Test sets

% Select which images will be in the training set and which will be in the 
% test set. Then randomly select 10 images of each individual specified and
% put them in the test set. The other images make up the training set.
SelectedIndexes = [];
for i = ppl-1
    dim = nnz(gnd == i+1);
    dimTot = nnz(gnd < i+1);
    SelectedIndexes = [SelectedIndexes, randperm(dim,10)+dimTot];
end
SelectedImages = zeros(length(gnd),1);
SelectedImages(SelectedIndexes) = 1;
SelectedImages = logical(SelectedImages);

TrainingFaces = fea(~SelectedImages,:)';
TestFaces = fea(SelectedImages,:)';

TrainingSize = size(TrainingFaces,2);
TestSize = size(TestFaces,2);

%% PCA

% Find the "mean face" and subtract it from all images in both sets.
MeanFace = sum(TrainingFaces,2)/TrainingSize;
CorrTrainingFaces = TrainingFaces - repmat(MeanFace,[1 TrainingSize]);
CorrTestFaces = TestFaces - repmat(MeanFace,[1 TestSize]);

[PCAEigVecs, PCAEigVals] = eig(CorrTrainingFaces*CorrTrainingFaces');

% Use the eigenvectors from above as a new basis, and transform the
% corrected training set into this bases, then cut unimportant dimensions.
PCATrainingFaces = diag(diag(PCAEigVals).^(-1/2))*PCAEigVecs'*CorrTrainingFaces;
PCATrainingFaces = flipud(PCATrainingFaces);
PCATrainingFaces = PCATrainingFaces(4:NumCoeffs,:);

% As above for the test set.
PCATestFaces = diag(diag(PCAEigVals).^(-1/2))*PCAEigVecs'*CorrTestFaces;
PCATestFaces = flipud(PCATestFaces);
PCATestFaces = PCATestFaces(4:NumCoeffs,:);

%% LDA

% Compute the within-class and between-class scatter matrices
ClassIndexes = gnd(~SelectedImages);
Classes = cell(38);
MeanClasses = cell(38);
WithinClasses = zeros(NumCoeffs-3);
BetweenClasses = zeros(NumCoeffs-3);
for i = 1:38
    Classes{i} = PCATrainingFaces(:,ClassIndexes == i);
    MeanClasses{i} = sum(Classes{i},2)/size(Classes{i},2);
    WithinClasses = WithinClasses + (Classes{i} - repmat(MeanClasses{i},[1 size(Classes{i},2)]))*(Classes{i} - repmat(MeanClasses{i},[1 size(Classes{i},2)]))';
    BetweenClasses = BetweenClasses + size(Classes{i},2)*MeanClasses{i}*MeanClasses{i}';
end

% Find the projection that maximises between-class spacing and
% minimises within-class spacing
[LDAEigVecs, ~] = eig(WithinClasses\BetweenClasses);
size(LDAEigVecs)
LDATrainingFaces = LDAEigVecs(:,1:37)'*PCATrainingFaces;
LDATestFaces = LDAEigVecs(:,1:37)'*PCATestFaces;

%% K-Nearest Neighbours Algorithm

% For each member of the test set, find the nearest members of the training
% set to determine the person represented
Successes = 0;
TotTrials = size(TestFaces,2);
for j = 1:TotTrials
    NearestIndexes = knnsearch(LDATrainingFaces',LDATestFaces(:,j)',...
        'K',NumNeighbours,...
        'Distance','cityblock');

    NearestImages = TrainingFaces(:,NearestIndexes);

    vec =  gnd(SelectedImages);
    vec2 = gnd(~SelectedImages);
    if nnz(vec2(NearestIndexes) == vec(j)) >= ceil(NumNeighbours/2)
        Successes = Successes + 1;
    end
    
    % Plot the first result for visualisation purposes
    if j == 1
        for i = 1:NumNeighbours
        subplot(2,NumNeighbours,i+NumNeighbours)
        imshow(uint8(reshape(NearestImages(:,i),32,32)))
        end
        subplot(2,NumNeighbours,ceil(NumNeighbours/2))
        imshow(uint8(reshape(TestFaces(:,1),32,32)))
    end
end

% Calculate the percentage of success rate
s = Successes/TotTrials*100;
fprintf("We've got %2.1f %% of accuracy!\n", s);