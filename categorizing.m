% Increased, Decreased, Neutral
label = zeros(1, neuron_data_size(2) - 1);
increased = double.empty;
decreased = double.empty;
neutral = double.empty;

for label_it = 1:neuron_data_size(2) - 1
    if behavior_data_correlation_factors(1, label_it) > sorted(round(shuffle_size * 0.95), label_it)
        label(1, label_it) = 1;
        increased = [increased og_neuron_data(:, label_it + 1)];
    else if behavior_data_correlation_factors(1, label_it) < sorted(round(shuffle_size * 0.05), label_it)
        label(1, label_it) = -1;
        decreased = [decreased og_neuron_data(:, label_it + 1)];
    else
        neutral = [neutral og_neuron_data(:, label_it + 1)];
    end
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
