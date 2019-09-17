clc
close all
clear all

T_Filename = input('behavior data file (format: ''filename.xlsx'') : ');
N_Filename = input('cell trace data file (format: ''filename.xlsx'') : ');
time_shift = 0;

% Create raw data matrix
tagged_data = xlsread(T_Filename);
neuron_data = xlsread(N_Filename);
[~, ~, raw] = xlsread(T_Filename);

% Correct the time shift
tagged_data = tagged_data-time_shift;

% Create positive and negative neuron data for positive and negative
% correlation
Positive_neuron_data = neuron_data;
Positive_neuron_data(Positive_neuron_data <= 0) = 0;
Negative_neuron_data = neuron_data;
Negative_neuron_data(Negative_neuron_data >= 0) = 0;

% Store the sizes of the data
tagged_data_size = size(tagged_data);
neuron_data_size = size(neuron_data);
    

% print behavior option menu
behavior_size = tagged_data_size(2)/2;
fprintf('\nBEHAVIOR OPTIONS: \n');
for behavior_it = 1:behavior_size
    fprintf(int2str(behavior_it) + ". " + convertCharsToStrings(raw(1, 2*behavior_it-1)) + '\n');
end

% receive input for which behavior to analyze
behavior = input('\nbehavior to analyze (format: #) : ');
behavior_fullname = raw(1, 2*behavior-1);
behavior_name=convertCharsToStrings(behavior_fullname);
fprintf("\nSelected Behavior: " + behavior_fullname + '\n');

r = 2*behavior-1;

% Creating empty array of the same size as the Behavior data
behavior_data  = zeros(neuron_data_size(1), neuron_data_size(2));

% Create an array complete time data increments from neuron data
Time = neuron_data(:, 1);
Time = Time - Time(1);

% Initialize arrays that store the data point numbers for Rearing or MB
% start and stop
behavior_data_size = find(tagged_data(:, r) > 0, 1, 'last');

if isempty(behavior_data_size) 
    behavior_data_size = 0; end

Behavior_start_stop = zeros(tagged_data_size(1), 2);

% Initialize an array, the same size as the neuron data, that stores
% increasing integer values for each behavior window
epochs = zeros(neuron_data_size(1), 1);
Behaviori = 1;

for iterator1 = 1:tagged_data_size(1)
    Behavior_start = find(Time >= tagged_data(iterator1, r), 1, 'first');
    Behavior_stop = find(Time >= tagged_data(iterator1, r+1), 1, 'first');
    
     % Records the stop and start data points
    if (iterator1 <= behavior_data_size && ~isempty(Behavior_start) && ~isempty(Behavior_stop) && Behavior_start~=1 && Behavior_stop ~=1)
        Behavior_start_stop(Behaviori, 1) = Behavior_start;
        Behavior_start_stop(Behaviori, 2) = Behavior_stop;
        Behaviori = Behaviori+1;
    end
    
    % Fills the cells with 1s in the given time interval
    behavior_data(Behavior_start:Behavior_stop, 1:neuron_data_size(2)) = 1; 
    
    % Fills the 'epochs' array with the intended values
    epochs(Behavior_start:Behavior_stop, 1) = iterator1;
    
end

behavior_data_size = find(Behavior_start_stop(:,1) > 0, 1, 'last');

if isempty(behavior_data_size)
    behavior_data_size = 0; 
end

Behavior_start_stop(behavior_data_size+1:end,:) = [];

% Plot all neuronal activity
% Plot all the neuron data channels on the same graph, normalized and separated
figure (1)
Normalizer = max(max(neuron_data(:,2:end)));
plot((Time - Time(1)), neuron_data(:, 2)/Normalizer + 1, 'Color', 'k', 'LineWidth', 1)
hold on
for channels = 3:neuron_data_size(2)
    plot((Time - Time(1)), neuron_data(:, channels)/Normalizer + channels - 1, 'Color', 'k', 'LineWidth', 1)
end

% Set axis size based on size of data
axis([0 (Time(end) - Time(1)) 0 neuron_data_size(2)])

% % Fill in the areas on the graph (Green --> MB, Red --> Rearing)
% area((Time - Time(1)), MB_data(:, 1)*neuron_data_size(2),'FaceColor','g', 'FaceAlpha',.2,'EdgeAlpha', 0.001)
% area((Time - Time(1)), Rearing_data(:, 1)*neuron_data_size(2),'FaceColor','r', 'FaceAlpha',.2,'EdgeAlpha', 0.001)
title('Raw Data') 
xlabel('Time (sec)')
ylabel('Neuron Channels')


% Plot behavior time area on neuronal activity
f1 = gca; % Get rescent figure properties(figure(1))

f1Behavior = figure(101);
ax1Behavior = axes;
copyobj(findobj(f1,'visible','on','type','line'), ax1Behavior);
behavior_title = sprintf('%s Raw data', behavior_name);
title(behavior_title)
% title(behavior_fullname + "(Raw Data)")
hold on;

for i = 1:size(Behavior_start_stop,1)
    t1 = Time(Behavior_start_stop(i,1));
    
    t2 = Time(Behavior_start_stop(i,2));
    area([t1 t2], [neuron_data_size(2)  neuron_data_size(2)],...
        'EdgeColor','none','FaceColor','b', 'FaceAlpha',0.5);
end
hold off;

%% Filter - Online deconvolution
% Filter the neuron data using Fast Online Deconvolution of Calcium Imaging Data

% User need to install or setup to run this code
% Friedrich-deconvolveCa: https://github.com/zhoupc/OASIS_matlab
% go to sub directory called OASIS and run ">> setup" if it shows "Error in
% deconvolveCa (line 70) options.sn = GetSn(y);"

filtered_data = zeros(neuron_data_size(1), 2*(neuron_data_size(2) - 1));
for iterator2 = 1:(neuron_data_size(2) - 1)
   [filtered_data(:, iterator2*2 - 1), filtered_data(:, iterator2*2)] = ...
       deconvolveCa(neuron_data(:, iterator2 + 1), 'ar1', 'foopsi', 'optimize_pars');
end

%Plot the filtered data just like the original data
figure (2)
Normalizer = max(max(filtered_data(:, 1:2:end)));
plot((Time - Time(1)), filtered_data(:, 1)/Normalizer + 1, 'Color', 'k', 'LineWidth', 1)
hold on
for iterator3 = 2:(neuron_data_size(2) - 1)
   corrected_filtered_data = filtered_data(:, iterator3*2 - 1)/Normalizer + iterator3;
   plot((Time - Time(1)),corrected_filtered_data,'Color', 'k', 'LineWidth', 1) 
end

% Set axis size based on size of data
axis([0 (Time(end) - Time(1)) 0 neuron_data_size(2)])

xlabel('Time (sec)')
ylabel('Neuron Channels')

% Plot behavior time area on neuronal activity
f2 = gca; % Get rescent figure properties(figure(1))

f2Behavior = figure(201);
ax2Behavior = axes;
copyobj(findobj(f2,'visible','on','type','line'), ax2Behavior);
title(behavior_fullname)
hold on;
for i = 1:size(Behavior_start_stop,1)
    t1 = Time(Behavior_start_stop(i,1));
    t2 = Time(Behavior_start_stop(i,2));
    area([t1 t2], [neuron_data_size(2)  neuron_data_size(2)],...
        'EdgeColor','none','FaceColor','b', 'FaceAlpha',0.5);
end

axis([0 (Time(end) - Time(1)) 0 neuron_data_size(2)])
hold off;

%% ROC
% Compute and plot the ROC curves for the orignal and filtered data

% USER INPUT

N_cell_number = input('Cell number : '); % iput cell number here to plot ROC curve (cell number ends at neuron_data_size(2) "34"
filter = input('0-original data, 1-filtered data : '); % 0: no filter, 1: filter
shuffle_size = input('shuffle size : ');
hist_bin_num = input('histogram bin number : ');

%  INITIALIZATION
if filter == 0
    wanted_neuron_data = neuron_data;
else
    wanted_neuron_data = filtered_data;
end

% Create arrays to store the auROC data
auROC_neuron_data = zeros(1, neuron_data_size(2) - 1);
auROC_raw_data = zeros(1, neuron_data_size(2) - 1);
auROC_filtered_data = zeros(1, neuron_data_size(2) - 1);

% neuron_data_size(2) looping for cells
for iterator4 = 2:neuron_data_size(2)
    [~, ~, ~, auROC_raw_data(1, iterator4 - 1)]...
        = perfcurve(behavior_data(:, 1), neuron_data(:, iterator4), 1);
    
    [~, ~, ~, auROC_filtered_data(1, iterator4 - 1)]...
        = perfcurve(behavior_data(:, 1), filtered_data(:, (iterator4 - 1)*2 - 1), 1);
end
if filter == 0
    auROC_neuron_data = auROC_raw_data;
else
    auROC_neuron_data = auROC_filtered_data;
end

% ROC plot
figure(999)
if filter == 0
    [x, y, ~, auROC_neuron_data(1, N_cell_number - 1)]...
        = perfcurve(behavior_data(:, 1), neuron_data(:, N_cell_number), 1);
     plot(x,y)
else
    [x, y, ~, auROC_neuron_data(1, N_cell_number - 1)]...
        = perfcurve(behavior_data(:, 1), filtered_data(:, (N_cell_number - 1)*2 - 1), 1);
     plot(x,y)
end
name = sprintf('%s ROC', behavior_name);
title(name)
xlabel('False Positive Rate')
ylabel('True Positive Rate')

% Calculate correlation factors
behavior_data_correlation_factors_raw = (auROC_raw_data - 0.5)*2;
behavior_data_correlation_factors_filtered = (auROC_filtered_data - 0.5)*2;
behavior_data_correlation_factors = (auROC_neuron_data - 0.5)*2;


% Store actual RS value of a specific cell
RS_actual = behavior_data_correlation_factors(1, N_cell_number);

% Shuffled ROC histogram calc/plot

shuffled_auROC_neuron_data = zeros(shuffle_size, neuron_data_size(2) - 1);

original = zeros(neuron_data_size(2),1);
shuffled = zeros(neuron_data_size(2),1);

for shuffle_it = 2:neuron_data_size(2)
    for random_it = 1:shuffle_size
        original = wanted_neuron_data(:,shuffle_it);
        shuffled = original(randperm(size(original,1)),:);
        
        [~, ~, ~, shuffled_auROC_neuron_data(random_it, shuffle_it - 1)]...
            = perfcurve(behavior_data(:, 1), shuffled, 1);
    end
end

% convert auROC --> RS
shuffled_RS_neuron_data = (shuffled_auROC_neuron_data - 0.5) * 2;

sorted = sort(shuffled_RS_neuron_data);  % sort RS data in ascending order

lowerlimit = sorted(round(shuffle_size*0.05), N_cell_number);
upperlimit = sorted(round(shuffle_size*0.95), N_cell_number);

% histogram
bellcurve = histfit(shuffled_RS_neuron_data(:,N_cell_number), hist_bin_num);
set(bellcurve(2),'color','k');
xlim([-0.6 0.6]);
ylim = get(gca,'YLim');
line([lowerlimit, lowerlimit], [0, ylim(2)], 'LineStyle', '--', 'Color', 'g')
line([upperlimit, upperlimit], [0, ylim(2)], 'LineStyle', '--', 'Color', 'g')
line([RS_actual, RS_actual], [0, ylim(2)], 'Color', 'r', 'LineWidth', 1)


% stem graph
figure(3)
stem(behavior_data_correlation_factors_raw)
hold on
stem(behavior_data_correlation_factors_filtered)

name = sprintf('%s Response Strengths', behavior_name);
title(name)
xlabel('Neuron Channels')
ylabel('Response Strength')
legend('Raw Data', 'Filtered Data')
axis([0 (neuron_data_size(2)-1) -1 1])

%% Response strengths of filtered data with bar graph

figure(11)
p = behavior_data_correlation_factors; p(p<=0) = 0; %only positive data
n = behavior_data_correlation_factors; n(n>0) = 0; % only negative data

bar(p, 'r'); hold on;  %positive data in red bar
bar(n, 'blue'); % negetive data in blue bar

title(name)
xlabel('Neuron Channels')
ylabel('Response Strength')
axis([0 (neuron_data_size(2)-1) -1 1])
hold off;

% Increased, Decreased, Neutral
label = zeros(1, neuron_data_size(2) - 1);
increased = double.empty;
decreased = double.empty;
neutral = double.empty;

for label_it = 1:neuron_data_size(2) - 1
    if behavior_data_correlation_factors(1, label_it) > sorted(round(shuffle_size * 0.95), label_it)
        label(1, label_it) = 1;
        increased = [increased neuron_data(:, label_it + 1)];
    elseif behavior_data_correlation_factors(1, label_it) < sorted(round(shuffle_size * 0.05), label_it)
        label(1, label_it) = -1;
        decreased = [decreased neuron_data(:, label_it + 1)];
    else
        neutral = [neutral neuron_data(:, label_it + 1)];
    end
end

image(increased');
colorbar;
colormap(jet(256));
title('Increased');

image(decreased');
colorbar;
colormap(jet(256));
title('Decreased');

image(neutral');
colorbar;
colormap(jet(256));
title('Neutral');


% Pie Chart
inc_size = size(increased);
dec_size = size(decreased);
neu_size = size(neutral);
X = [inc_size(2), neu_size(2), dec_size(2)];
labels = {'increased', 'neutral', 'decreased'};
pie(X, labels);

figure(987)
p = behavior_data_correlation_factors; p(p<=0) = 0; %only positive data
n = behavior_data_correlation_factors; n(n>0) = 0; % only negative data

bar(p, 'r'); hold on;  %positive data in red bar
bar(n, 'blue'); % negetive data in blue bar

title(behavior_name + " Response Strengths")
xlabel('Neuron Channels')
ylabel('Response Strength')
axis([0 (neuron_data_size(2)-1) -1 1])
hold off;

% heatmap

% cell_activity = neuron_data(:, 2:neuron_data_size(2));
% cell_activity = filtered_data;
% cell_activity = increased;
% cell_activity = decreased;
cell_activity = neutral;
interval = 100; % 5 sec
cell_activity_size = size(cell_activity);

Behavior_start_stop_size = size(Behavior_start_stop);
heatmap_data_start = zeros(interval*2+1, cell_activity_size(2));
heatmap_data_stop = zeros(interval*2+1, cell_activity_size(2));
for heatmap_it = 1:Behavior_start_stop_size(1)
    start = Behavior_start_stop(heatmap_it, 1);
    stop = Behavior_start_stop(heatmap_it, 2);
    heatmap_data_start = heatmap_data_start + cell_activity([start-interval:start+interval],:);
    heatmap_data_stop = heatmap_data_stop + cell_activity([stop-interval:stop+interval],:);
end
heatmap_data_start = heatmap_data_start / Behavior_start_stop_size(1);
heatmap_data_stop = heatmap_data_stop / Behavior_start_stop_size(1);
image(heatmap_data_start');
colorbar;
colormap(jet(256));
ylim = get(gca,'YLim');
line([interval, interval], [0, ylim(2)], 'Color', 'k', 'LineWidth', 2)
title('Start');

image(heatmap_data_stop');
colorbar;
colormap(jet(256));
title('Stop');
ylim = get(gca,'YLim');
line([interval, interval], [0, ylim(2)], 'Color', 'k', 'LineWidth', 2)


% heatmap sorting
sum_before = zeros(1, cell_activity_size(2));
sum_after = zeros(1, cell_activity_size(2));

for score_it = 1:interval
    sum_before = sum_before + heatmap_data_start(score_it, :);
    sum_after = sum_after + heatmap_data_start(score_it+interval, :);
end

scores = sum_before + sum_after; %heatmap_data_start(100, :);
score_map = [scores; heatmap_data_start];
sorted_heatmap = sortrows(score_map', 1, 'descend');
image(sorted_heatmap);
colorbar;
colormap(jet(256));
ylim = get(gca,'YLim');
line([interval, interval], [0, ylim(2)], 'Color', 'k', 'LineWidth', 2)
title('Sorted Heatmap');

% Label_list = {(G_data(:, 1) + 2*D_data(:, 1)),...
%                 (2*D_data(:, 1)+3*R_data(:,1)),...
%                 G_data(:,1), D_data(:,1), R_data(:,1)};

Label_list = {(G_data(:,1)+2*D_data(:,1)+5*R_data(:,1))};

for  iterator_all = 1:length(Label_list)
    %% LDA
    % Apply Linear Discriminant Analysis to the neuron data. Then scatter plot
    % the neuron channels based on the eigenvectors determined from the LDA
    
    % Creating an array of simply the denoised filtered data
    filtered = 1;
    if(filtered)
        lda_data = filtered_data;
        lda_data(:, 2:2:end) = []; 
    else
        lda_data = neuron_data(:, 2:end);
    end
    % Creating the "Labels" for the LDA (0->no behavior, 1->Grooming, 2->Digging, 3->Rearing)
    % Labels = (G_data(:, 1) + 2*D_data(:, 1)+3*R_data(:,1));
    % Labels = (G_data(:, 1) + 2*D_data(:, 1));
    % Labels = (2*D_data(:, 1)+3*R_data(:,1));
    % Labels = (G_data(:, 1));
    % Labels = (2*D_data(:, 1));
    % Labels = (3*R_data(:,1));
    % Labels(Labels == 3 && Labels == 4 ) = 0;
    Labels = Label_list{iterator_all};

    %% Apply the built in MATLAB LDA 
    % Creates Figures 6 and 7 which are visualizations of the optimization
    % parameters
    Mdl = fitcdiscr(lda_data, Labels, 'OptimizeHyperparameters','auto', 'DiscrimType', 'pseudolinear');
    
    % Determine the eigenvector and corresponding eigenvalues from the LDA
    % model. Then sort the eigenvalues to find the two highest values and use
    % the corresponding eigenvectors for the LDA plot
    % [V,Lambda] = eig(Mdl.BetweenSigma, Mdl.Sigma, 'qz'); %Computes eigenvectors in matrix V and eigenvalues in matrix 
    [V,Lambda] = eig(Mdl.Sigma\Mdl.BetweenSigma);
    [Lambda, sorted] = sort(diag(Lambda), 'descend');
    
    
    %Scatter plot the eigenvectors with the hightest corresponding eigenvalues
    figure (7)
    scatter(V(:, sorted(1)), V(:, sorted(2)))
    
    for iterator5 = 1:neuron_data_size(2) - 1
        text(V(iterator5, sorted(1)), V(iterator5, sorted(2)), num2str(iterator5)) 
    end
    
    title('Linear Discriminant Analysis Plot (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    axis([-1 1 -1 1])

    %% Plot LDA by groups
    % Plot the LDA by groups relative to Grooming, Digging, Rearing correlation factors
    correlation_groups = zeros(neuron_data_size(2) - 1, 7);
    for iterator6 = 1:neuron_data_size(2) - 1
       if (G_filtered_data_correlation_factors(iterator6) > 0.1)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 20;
           
           correlation_groups(iterator6, 2:4) = correlation_groups(iterator6, 2:4)...
               + 100*G_filtered_data_correlation_factors(iterator6);
       end
       if (D_filtered_data_correlation_factors(iterator6) > 0.1)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 5;
           
          correlation_groups(iterator6, 4:6) = correlation_groups(iterator6, 4:6)...
               + 100*D_filtered_data_correlation_factors(iterator6);
       end
       
       if (R_filtered_data_correlation_factors(iterator6) > 0.1)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 1;
           
          correlation_groups(iterator6, 3) = correlation_groups(iterator6, 3)...
               + 100*R_filtered_data_correlation_factors(iterator6);
          correlation_groups(iterator6, 5) = correlation_groups(iterator6, 5)...
               + 100*R_filtered_data_correlation_factors(iterator6);
          correlation_groups(iterator6, 7) = correlation_groups(iterator6, 7)...
               + 100*R_filtered_data_correlation_factors(iterator6);
       end
    end
       
    
    G = find(correlation_groups(:, 1) == 20);
    D = find(correlation_groups(:, 1) == 5); 
    R = find(correlation_groups(:, 1) == 1); 
    GD = find(correlation_groups(:, 1) == 25);
    DR = find(correlation_groups(:, 1) == 6); 
    RG = find(correlation_groups(:, 1) == 21); 
    baseline = find(correlation_groups(:, 1) == 0); 
    
    
    correlation_groups(:,2:7) = correlation_groups(:,2:7) - min(min(correlation_groups(:,2:7)));
    
    figure
        b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
%     scatter(V(:, sorted(1)), V(:, sorted(2)), 'MarkerFacecolor', 'k')
    hold on;
    scatter(V(baseline, sorted(1)), V(baseline, sorted(2)), mean(mean(correlation_groups(:,2:7))), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', 'k')
    
    axis([-0.4 0.5 -0.4 0.8])
    hold on;
    s1 = scatter(V([D; GD;DR], sorted(1)), V([D;GD;DR], sorted(2)), correlation_groups([D;GD;DR], 6), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y,'MarkerFaceAlpha',.5);
    fitellipse(V([D; GD;DR], sorted(1)), V([D;GD;DR], sorted(2)), y);
    s2 = scatter(V([R;DR; RG], sorted(1)), V([R; DR; RG], sorted(2)), correlation_groups([R; DR; RG], 7), 'filled', ...
        'MarkerEdgeColor', b,'MarkerFaceColor', b,'MarkerFaceAlpha',.5);
    fitellipse(V([R;DR; RG], sorted(1)), V([R; DR; RG], sorted(2)), b);
    s3 = scatter(V([G; GD; RG], sorted(1)), V([G; GD; RG], sorted(2)), correlation_groups([G; GD; RG], 2),'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r,'MarkerFaceAlpha',.5);
    fitellipse(V([G; GD; RG], sorted(1)), V([G; GD; RG], sorted(2)), r)
    
%     title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    axis([-0.4 0.5 -0.4 0.8])
%     legend([s1 s2 s3], {'Digging', 'Rearing', 'Grooming'})
    
    f1 = gca;
    saveas(f1,[num2str(iterator_all) '_test1.jpg'])
    hold off; 
    

    %% Plot LDA by groups
    % Plot the LDA by groups relative to Grooming, Digging correlation factors
    correlation_groups = zeros(neuron_data_size(2) - 1, 5);
    for iterator6 = 1:neuron_data_size(2) - 1
       if (G_filtered_data_correlation_factors(iterator6) > 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 20;
           
           correlation_groups(iterator6, 2:3) = correlation_groups(iterator6, 2:3)...
               + 100*G_filtered_data_correlation_factors(iterator6);
       end
       if (G_filtered_data_correlation_factors(iterator6) <= 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 1;
           
           correlation_groups(iterator6, 4:5) = correlation_groups(iterator6, 4:5)...
               - 100*G_filtered_data_correlation_factors(iterator6);
       end
       
       if (D_filtered_data_correlation_factors(iterator6) > 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 5;
           
          correlation_groups(iterator6, 3) = correlation_groups(iterator6, 3)...
               + 100*D_filtered_data_correlation_factors(iterator6);
          correlation_groups(iterator6, 5) = correlation_groups(iterator6, 5)...
               + 100*D_filtered_data_correlation_factors(iterator6);
       end
       if (D_filtered_data_correlation_factors(iterator6) <= 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 10;
           
           correlation_groups(iterator6, 2) = correlation_groups(iterator6, 2)...
               - 100*D_filtered_data_correlation_factors(iterator6);
           correlation_groups(iterator6, 4) = correlation_groups(iterator6, 4)...
               - 100*D_filtered_data_correlation_factors(iterator6);
       end
    end
       
    
    N_G__P_D = find(correlation_groups(:, 1) == 6); length(N_G__P_D)
    N_G__N_D = find(correlation_groups(:, 1) == 11); length(N_G__N_D)
    P_G__P_D = find(correlation_groups(:, 1) == 25); length(P_G__P_D)
    P_G__N_D = find(correlation_groups(:, 1) == 30); length(P_G__N_D)
    
    correlation_groups(:,2:5) = correlation_groups(:,2:5) - min(min(correlation_groups(:,2:5)));
    
    figure
    % gscatter(V(:, sorted(1)), V(:, sorted(2)), correlation_groups)
    
        b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
    scatter(V(N_G__N_D, sorted(1)), V(N_G__N_D, sorted(2)), correlation_groups(N_G__N_D, 5), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', g)
    hold on
    scatter(V(N_G__P_D, sorted(1)), V(N_G__P_D, sorted(2)), correlation_groups(N_G__P_D, 4), 'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r)
    scatter(V(P_G__P_D, sorted(1)), V(P_G__P_D, sorted(2)), correlation_groups(P_G__P_D, 3), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y)
    scatter(V(P_G__N_D, sorted(1)), V(P_G__N_D, sorted(2)), correlation_groups(P_G__N_D, 2), 'filled', ...
        'MarkerEdgeColor', b,'MarkerFaceColor', b)
    
    title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    axis([-0.4 0.7 -0.4 0.7])
    legend('baseline', 'D↑', 'D↑G↑', 'G↑')
    
    f1 = gca;
    saveas(f1,[num2str(iterator_all) '_test1.jpg'])
    hold off;
    

    %% Plot LDA by groups
    % Plot the LDA by groups relative to Digging, Rearing correlation factors
    correlation_groups = zeros(neuron_data_size(2) - 1, 5);
    for iterator6 = 1:neuron_data_size(2) - 1
       if (D_filtered_data_correlation_factors(iterator6) > 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 20;
           
           correlation_groups(iterator6, 2:3) = correlation_groups(iterator6, 2:3)...
               + 100*D_filtered_data_correlation_factors(iterator6);
       end
       if (D_filtered_data_correlation_factors(iterator6) <= 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 1;
           
           correlation_groups(iterator6, 4:5) = correlation_groups(iterator6, 4:5)...
               - 100*D_filtered_data_correlation_factors(iterator6);
       end
       
       if (R_filtered_data_correlation_factors(iterator6) > 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 5;
           
          correlation_groups(iterator6, 3) = correlation_groups(iterator6, 3)...
               + 100*R_filtered_data_correlation_factors(iterator6);
          correlation_groups(iterator6, 5) = correlation_groups(iterator6, 5)...
               + 100*R_filtered_data_correlation_factors(iterator6);
       end
       if (R_filtered_data_correlation_factors(iterator6) <= 0)
           correlation_groups(iterator6, 1) = correlation_groups(iterator6) + 10;
           
           correlation_groups(iterator6, 2) = correlation_groups(iterator6, 2)...
               - 100*R_filtered_data_correlation_factors(iterator6);
           correlation_groups(iterator6, 4) = correlation_groups(iterator6, 4)...
               - 100*R_filtered_data_correlation_factors(iterator6);
       end
    end
       
    
    N_D__P_R = find(correlation_groups(:, 1) == 6); length(N_D__P_R)
    N_D__N_R = find(correlation_groups(:, 1) == 11); length(N_D__N_R)
    P_D__P_R = find(correlation_groups(:, 1) == 25); length(P_D__P_R)
    P_D__N_R = find(correlation_groups(:, 1) == 30); length(P_D__N_R)
    
    correlation_groups(:,2:5) = correlation_groups(:,2:5) - min(min(correlation_groups(:,2:5)));
    
    figure
    % gscatter(V(:, sorted(1)), V(:, sorted(2)), correlation_groups)
    
        b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
    scatter(V(N_D__N_R, sorted(1)), V(N_D__N_R, sorted(2)), correlation_groups(N_D__N_R, 5), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', g)
    hold on
    scatter(V(N_D__P_R, sorted(1)), V(N_D__P_R, sorted(2)), correlation_groups(N_D__P_R, 4), 'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r)
    scatter(V(P_D__P_R, sorted(1)), V(P_D__P_R, sorted(2)), correlation_groups(P_D__P_R, 3), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y)
    scatter(V(P_D__N_R, sorted(1)), V(P_D__N_R, sorted(2)), correlation_groups(P_D__N_R, 2), 'filled', ...
        'MarkerEdgeColor', b,'MarkerFaceColor', b)
    
    title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    axis([-0.4 0.7 -0.4 0.7])
    legend('baseline', 'R↑', 'D↑R↑', 'D↑')
    
    f2 = gca;
    saveas(f2,[num2str(iterator_all) '_test2.jpg'])
    hold off;
    

    %%LDA in one Group
    
    correlation_groups2 = zeros(neuron_data_size(2) - 1, 4);
    for iterator6 = 1:neuron_data_size(2) - 1
       if (G_filtered_data_correlation_factors(iterator6) > 0.1)
           correlation_groups2(iterator6, 1) = correlation_groups2(iterator6) + 1;
           
           correlation_groups2(iterator6, 2) = correlation_groups2(iterator6, 2)...
               + 100*G_filtered_data_correlation_factors(iterator6);
       end
       if (G_filtered_data_correlation_factors(iterator6) <= 0.1 && ...
               G_filtered_data_correlation_factors(iterator6) > -0.1)
           correlation_groups2(iterator6, 1) = correlation_groups2(iterator6) + 2;
           
           correlation_groups2(iterator6, 3) = correlation_groups2(iterator6, 3)...
               + 100*abs(G_filtered_data_correlation_factors(iterator6));
       end
       
       if (G_filtered_data_correlation_factors(iterator6) < -0.1)
           correlation_groups2(iterator6, 1) = correlation_groups2(iterator6) + 3;
           
          correlation_groups2(iterator6, 4) = correlation_groups2(iterator6, 4)...
               + 100*abs(G_filtered_data_correlation_factors(iterator6));
       end
    end
       
    
    P_G = find(correlation_groups2(:, 1) == 1); length(P_G) % Positive correl
    B_G = find(correlation_groups2(:, 1) == 2); length(B_G) % less than 10% base line
    N_G = find(correlation_groups2(:, 1) == 3); length(N_G) % Negative correl
    
    
    figure
    % gscatter(V(:, sorted(1)), V(:, sorted(2)), correlation_groups2)
    
%         b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
    scatter(V(P_G, sorted(1)), V(P_G, sorted(2)), correlation_groups2(P_G, 2), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', g)
    hold on
    scatter(V(B_G, sorted(1)), V(B_G, sorted(2)), correlation_groups2(B_G, 3), 'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r)
    scatter(V(N_G, sorted(1)), V(N_G, sorted(2)), correlation_groups2(N_G, 4), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y)
    
    
    title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    legend('G↑', 'baseline', 'G↓')
    axis([-0.4 0.7 -0.4 0.7])
    
    f3 = gca;
    saveas(f3,[num2str(iterator_all) '_test3.jpg'])

    %%LDA in Digging Group
    
    correlation_groups3 = zeros(neuron_data_size(2) - 1, 4);
    for iterator8 = 1:neuron_data_size(2) - 1
       if (D_filtered_data_correlation_factors(iterator8) > 0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 1;
           
           correlation_groups3(iterator8, 2) = correlation_groups3(iterator8, 2)...
               + 100*D_filtered_data_correlation_factors(iterator8);
       end
       if (D_filtered_data_correlation_factors(iterator8) <= 0.1 && ...
               D_filtered_data_correlation_factors(iterator8) > -0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 2;
           
           correlation_groups3(iterator8, 3) = correlation_groups3(iterator8, 3)...
               + 100*abs(D_filtered_data_correlation_factors(iterator8));
       end
       
       if (D_filtered_data_correlation_factors(iterator8) < -0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 3;
           
          correlation_groups3(iterator8, 4) = correlation_groups3(iterator8, 4)...
               + 100*abs(D_filtered_data_correlation_factors(iterator8));
       end
    end
       
    
    P_D = find(correlation_groups3(:, 1) == 1); length(P_D) % Positive correl
    B_D = find(correlation_groups3(:, 1) == 2); length(B_D) % less than 10% base line
    N_D = find(correlation_groups3(:, 1) == 3); length(N_D) % Negative correl
    
    
    figure
    % gscatter(V(:, sorted(1)), V(:, sorted(2)), correlation_groups2)
    
%         b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
    scatter(V(P_D, sorted(1)), V(P_D, sorted(2)), correlation_groups3(P_D, 2), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', g)
    hold on
    scatter(V(B_D, sorted(1)), V(B_D, sorted(2)), correlation_groups3(B_D, 3), 'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r)
    scatter(V(N_D, sorted(1)), V(N_D, sorted(2)), correlation_groups3(N_D, 4), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y)
    
    
    title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    legend('D↑', 'baseline', 'D↓')
    axis([-0.4 0.7 -0.4 0.7])
    
    f4 = gca;
    saveas(f4,[num2str(iterator_all) '_test4.jpg'])

    %%LDA in Digging Group
    
    correlation_groups3 = zeros(neuron_data_size(2) - 1, 4);
    for iterator8 = 1:neuron_data_size(2) - 1
       if (R_filtered_data_correlation_factors(iterator8) > 0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 1;
           
           correlation_groups3(iterator8, 2) = correlation_groups3(iterator8, 2)...
               + 100*R_filtered_data_correlation_factors(iterator8);
       end
       if (R_filtered_data_correlation_factors(iterator8) <= 0.1 && ...
               R_filtered_data_correlation_factors(iterator8) > -0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 2;
           
           correlation_groups3(iterator8, 3) = correlation_groups3(iterator8, 3)...
               + 100*abs(R_filtered_data_correlation_factors(iterator8));
       end
       
       if (R_filtered_data_correlation_factors(iterator8) < -0.1)
           correlation_groups3(iterator8, 1) = correlation_groups3(iterator8) + 3;
           
          correlation_groups3(iterator8, 4) = correlation_groups3(iterator8, 4)...
               + 100*abs(R_filtered_data_correlation_factors(iterator8));
       end
    end
       
    
    P_R = find(correlation_groups3(:, 1) == 1); length(P_R) % Positive correl
    B_R = find(correlation_groups3(:, 1) == 2); length(B_R) % less than 10% base line
    N_R = find(correlation_groups3(:, 1) == 3); length(N_R) % Negative correl
    
    
    figure
    % gscatter(V(:, sorted(1)), V(:, sorted(2)), correlation_groups2)
    
        b = [143/255 170/255 220/255];
        r = [220/255 96/255 42/255];
        y = [237 177 32]/255;
        g = [180 180 180]/255;
        
    scatter(V(P_R, sorted(1)), V(P_R, sorted(2)), correlation_groups3(P_R, 2), 'filled', ...
        'MarkerEdgeColor', g,'MarkerFaceColor', g)
    hold on
    scatter(V(B_R, sorted(1)), V(B_R, sorted(2)), correlation_groups3(B_R, 3), 'filled', ...
        'MarkerEdgeColor', r,'MarkerFaceColor', r)
    scatter(V(N_R, sorted(1)), V(N_R, sorted(2)), correlation_groups3(N_R, 4), 'filled', ...
        'MarkerEdgeColor', y,'MarkerFaceColor', y)
    
    
    title('LDA with Correlation Factors (Filtered Data)')
    xlabel('Linear Coordiate 1')
    ylabel('Linear Coordiate 2')
    legend('R↑', 'baseline', 'R↓')
    axis([-0.4 0.7 -0.4 0.7])
    
    f5 = gca;
    saveas(f5,[num2str(iterator_all) '_test5.jpg'])
end
