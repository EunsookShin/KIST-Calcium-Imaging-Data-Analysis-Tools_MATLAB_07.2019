% Shuffled ROC histogram calc/plot

shuffled_auROC_neuron_data = zeros(shuffle_size, neuron_data_size(2) - 1);

original = zeros(neuron_data_size(2),1);
shuffled = zeros(neuron_data_size(2),1);

for shuffle_it = 2:neuron_data_size(2)
    for random_it = 1:shuffle_size
        original = neuron_data(:,shuffle_it);
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
