% heatmap

cell_activity =  og_neuron_data(:, 2:neuron_data_size(2));
G_start_stop_size = size(G_start_stop);
heatmap_data_start = zeros(201, neuron_data_size(2) - 1);
heatmap_data_stop = zeros(201, neuron_data_size(2) - 1);
for heatmap_it = 1:G_start_stop_size(1)
    %figure(heatmap_it+300);
    start = G_start_stop(heatmap_it, 1);
    stop = G_start_stop(heatmap_it, 2);
    heatmap_data_start = heatmap_data_start + cell_activity([start-100:start+100],:);
    heatmap_data_stop = heatmap_data_stop + cell_activity([stop-100:stop+100],:);
end
heatmap_data_start = heatmap_data_start / G_start_stop_size(1);
heatmap_data_stop = heatmap_data_stop / G_start_stop_size(1);
image(heatmap_data_start');
colorbar;
colormap(jet(256));
title('Start');

image(heatmap_data_stop');
colorbar;
colormap(jet(256));
title('Stop');
