%% ROC
% Compute and plot the ROC curves for the orignal and filtered data

% USER INPUT
N_cell_number = 3; % iput cell number here to plot ROC curve (cell number ends at neuron_data_size(2) "34"
filter = 1; % 0: no filter, 1: filter
shuffle_size = 50;
hist_bin_num = 5;

%  INITIALIZATION
behavior_data = G_data;
if filter == 0
    neuron_data = og_neuron_data;
else
    neuron_data = filtered_data;
end

% Create arrays to store the auROC data
auROC_neuron_data = zeros(1, neuron_data_size(2) - 1);

% neuron_data_size(2) looping for cells
for iterator4 = 2:neuron_data_size(2)
    if filter == 0
        [~, ~, ~, auROC_neuron_data(1, iterator4 - 1)]...
            = perfcurve(behavior_data(:, 1), neuron_data(:, iterator4), 1);
    else
        [~, ~, ~, auROC_neuron_data(1, iterator4 - 1)]...
            = perfcurve(behavior_data(:, 1), neuron_data(:, (iterator4 - 1)*2 - 1), 1);
    end
end


% ROC plot
    figure(999)
    [x, y, ~, auROC_neuron_data(1, N_cell_number - 1)]...
        = perfcurve(behavior_data(:, 1), neuron_data(:, (N_cell_number - 1)*2 - 1), 1);
     plot(x,y)
  

% Calculate correlation factors
behavior_data_correlation_factors = (auROC_neuron_data - 0.5)*2;

% Store actual RS value of a specific cell
RS_actual = behavior_data_correlation_factors(1, N_cell_number);
