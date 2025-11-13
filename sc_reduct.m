%% Wind Power Scenario Reduction: FFS vs Backward Reduction
clear; close all; clc;

%% 1. Generate Synthetic Wind Power Scenarios
rng(42); % For reproducibility
num_original_scenarios = 200;
num_time_periods = 24; % 24-hour profile
num_reduced_scenarios = 20;

% Generate correlated wind power scenarios (mean ~0.5, some correlation)
base_profile = 0.4 + 0.2 * sin(2*pi*(0:num_time_periods-1)/24);
wind_scenarios = zeros(num_original_scenarios, num_time_periods);

for i = 1:num_original_scenarios
    % Add random variations with temporal correlation
    noise = 0.1 * randn(1, num_time_periods);
    for t = 2:num_time_periods
        noise(t) = 0.7 * noise(t-1) + 0.3 * noise(t);
    end
    wind_scenarios(i, :) = base_profile + noise;
    wind_scenarios(i, :) = max(0, min(1, wind_scenarios(i, :))); % Clip to [0,1]
end

% Equal probabilities for original scenarios
original_probs = ones(num_original_scenarios, 1) / num_original_scenarios;

fprintf('Generated %d wind power scenarios (%d time periods)\n', ...
        num_original_scenarios, num_time_periods);
fprintf('Target: Reduce to %d scenarios\n\n', num_reduced_scenarios);

%% 2. Distance Matrix Calculation
fprintf('Calculating distance matrix...\n');
distance_matrix = calculate_distance_matrix(wind_scenarios);

%% 3. Fast Forward Selection (FFS) Algorithm
fprintf('Running Fast Forward Selection...\n');
tic;
[ffs_scenarios, ffs_probs, ffs_indices] = fast_forward_selection(...
    wind_scenarios, original_probs, num_reduced_scenarios, distance_matrix);
ffs_time = toc;

%% 4. Backward Reduction Algorithm
fprintf('Running Backward Reduction...\n');
tic;
[backward_scenarios, backward_probs, backward_indices] = backward_reduction(...
    wind_scenarios, original_probs, num_reduced_scenarios, distance_matrix);
backward_time = toc;

%% 5. Performance Evaluation - CORRECTED VERSION
fprintf('\n=== PERFORMANCE COMPARISON ===\n');

% Calculate proper distance metrics
ffs_metrics = calculate_proper_metrics(wind_scenarios, original_probs, ...
                                     ffs_scenarios, ffs_probs);
backward_metrics = calculate_proper_metrics(wind_scenarios, original_probs, ...
                                          backward_scenarios, backward_probs);

fprintf('\nComputation Time:\n');
fprintf('  FFS:       %.3f seconds\n', ffs_time);
fprintf('  Backward:  %.3f seconds\n', backward_time);

fprintf('\nDistance Metrics Comparison:\n');
fprintf('  Metric                     FFS         Backward\n');
fprintf('  ----------------------------------------------\n');
fprintf('  Euclidean RMS              %-10.6f  %-10.6f\n', ...
        ffs_metrics.euclidean_rms, backward_metrics.euclidean_rms);
fprintf('  Manhattan RMS              %-10.6f  %-10.6f\n', ...
        ffs_metrics.manhattan_rms, backward_metrics.manhattan_rms);
fprintf('  Wasserstein (1st-order)    %-10.6f  %-10.6f\n', ...
        ffs_metrics.wasserstein_1, backward_metrics.wasserstein_1);
fprintf('  Kantorovich-Rubinstein     %-10.6f  %-10.6f\n', ...
        ffs_metrics.kantorovich, backward_metrics.kantorovich);
fprintf('  Variance Preserved         %-10.6f  %-10.6f\n', ...
        ffs_metrics.variance_preserved, backward_metrics.variance_preserved);

%% New Helper Function for Proper Distance Metrics
function metrics = calculate_proper_metrics(original_scenarios, original_probs, ...
                                          reduced_scenarios, reduced_probs)
    
    num_original = size(original_scenarios, 1);
    num_reduced = size(reduced_scenarios, 1);
    
    % 1. Euclidean Distance (Root Mean Square)
    euclidean_dists = zeros(num_original, 1);
    for i = 1:num_original
        min_dist = inf;
        for j = 1:num_reduced
            % Euclidean distance between scenarios
            dist = sqrt(sum((original_scenarios(i,:) - reduced_scenarios(j,:)).^2));
            min_dist = min(min_dist, dist);
        end
        euclidean_dists(i) = min_dist;
    end
    metrics.euclidean_rms = sqrt(mean(euclidean_dists.^2));
    
    % 2. Manhattan Distance (Root Mean Square)
    manhattan_dists = zeros(num_original, 1);
    for i = 1:num_original
        min_dist = inf;
        for j = 1:num_reduced
            % Manhattan distance between scenarios
            dist = sum(abs(original_scenarios(i,:) - reduced_scenarios(j,:)));
            min_dist = min(min_dist, dist);
        end
        manhattan_dists(i) = min_dist;
    end
    metrics.manhattan_rms = sqrt(mean(manhattan_dists.^2));
    
    % 3. Mahalanobis Distance (using original data covariance)
    sigma = cov(original_scenarios);
    % Add small regularization for numerical stability
    sigma_reg = sigma + 1e-6 * eye(size(sigma));
    try
        inv_sigma = inv(sigma_reg);
        mahalanobis_dists = zeros(num_original, 1);
        for i = 1:num_original
            min_dist = inf;
            for j = 1:num_reduced
                diff = (original_scenarios(i,:) - reduced_scenarios(j,:));
                dist = sqrt(diff * inv_sigma * diff');
                min_dist = min(min_dist, dist);
            end
            mahalanobis_dists(i) = min_dist;
        end
        metrics.mahalanobis_rms = sqrt(mean(mahalanobis_dists.^2));
    catch
        metrics.mahalanobis_rms = NaN;
    end
    
    % 4. 1st-order Wasserstein Distance (Earth Mover's Distance)
    % This is a simplified computation - full version would require linear programming
    wasserstein_dists = zeros(num_original, 1);
    for i = 1:num_original
        min_dist = inf;
        for j = 1:num_reduced
            % Using Euclidean as ground metric
            dist = norm(original_scenarios(i,:) - reduced_scenarios(j,:));
            min_dist = min(min_dist, dist);
        end
        wasserstein_dists(i) = min_dist;
    end
    metrics.wasserstein_1 = mean(wasserstein_dists .* original_probs);
    
    % 5. Kantorovich-Rubinstein Metric (approximation)
    % For Lipschitz functions with constant 1, this equals 1st-order Wasserstein
    metrics.kantorovich = metrics.wasserstein_1;
    
    % 6. Statistical Preservation
    original_var = trace(cov(original_scenarios));
    reduced_var = trace(weighted_cov(reduced_scenarios, reduced_probs));
    metrics.variance_preserved = reduced_var / original_var;
    
    % 7. Mean Preservation
    original_mean = mean(original_scenarios);
    reduced_mean = reduced_probs' * reduced_scenarios;
    metrics.mean_preservation_error = norm(original_mean - reduced_mean);
end


%% 6. Visualization
figure('Position', [100, 100, 1400, 900]);

% Subplot 1: Original vs Reduced Scenarios
figure(1);
plot(wind_scenarios', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;
plot(ffs_scenarios', 'r-', 'LineWidth', 2);
title('Original (gray) vs FFS Reduced (red)');
xlabel('Time (hours)'); ylabel('Wind Power (p.u.)');
grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;


figure(2);
plot(wind_scenarios', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;
plot(backward_scenarios', 'b-', 'LineWidth', 2);
title('Original (gray) vs Backward Reduced (blue)');
xlabel('Time (hours)'); ylabel('Wind Power (p.u.)');
grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

% Subplot 3: Mean Comparison
figure(3);
original_mean = mean(wind_scenarios);
ffs_mean = ffs_probs' * ffs_scenarios;
backward_mean = backward_probs' * backward_scenarios;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

plot(original_mean, 'k--', 'LineWidth', 2, 'DisplayName', 'Original Mean'); hold on;
plot(ffs_mean, 'r-', 'LineWidth', 2, 'DisplayName', 'FFS Mean');
plot(backward_mean, 'b-', 'LineWidth', 2, 'DisplayName', 'Backward Mean');
title('Mean Profile Comparison');
xlabel('Time (hours)'); ylabel('Wind Power (p.u.)');
legend; grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

% Subplot 4: Performance Metrics Comparison
figure(4);
metrics_plot = [ffs_metrics.euclidean_rms, backward_metrics.euclidean_rms;
               ffs_metrics.wasserstein_1, backward_metrics.wasserstein_1;
               ffs_metrics.mean_preservation_error, backward_metrics.mean_preservation_error;
               ffs_time, backward_time];
metric_names = {'Euclidean RMS', 'Wasserstein-1', 'Mean Error', 'Comp Time'};
bar(metrics_plot);
set(gca, 'XTickLabel', metric_names);
ylabel('Value');
title('Proper Distance Metrics Comparison');
legend('FFS', 'Backward', 'Location', 'northwest');
grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

% Subplot 5: Selected Scenarios Distribution
figure(5);
scatter(ffs_indices, ffs_probs * num_original_scenarios, 50, 'r', 'filled');
hold on;
scatter(backward_indices, backward_probs * num_original_scenarios, 50, 'b', 'filled');
xlabel('Original Scenario Index');
ylabel('Probability Weight (scaled)');
title('Selected Scenarios Distribution');
legend('FFS', 'Backward');
grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

% Subplot 6: Variance Preservation
figure(6);
original_var = var(wind_scenarios);
ffs_var = weighted_variance(ffs_scenarios, ffs_probs);
backward_var = weighted_variance(backward_scenarios, backward_probs);

plot(original_var, 'k--', 'LineWidth', 2, 'DisplayName', 'Original'); hold on;
plot(ffs_var, 'r-', 'LineWidth', 2, 'DisplayName', 'FFS');
plot(backward_var, 'b-', 'LineWidth', 2, 'DisplayName', 'Backward');
title('Variance Preservation');
xlabel('Time (hours)'); ylabel('Variance');
legend; grid on;
a=gca;
a.FontName='Times New Roman';
a.FontSize=32;

%% Helper Functions

function distance_matrix = calculate_distance_matrix(scenarios)
    % Calculate Euclidean distance matrix between all scenario pairs
    num_scenarios = size(scenarios, 1);
    distance_matrix = zeros(num_scenarios);
    
    for i = 1:num_scenarios
        for j = i+1:num_scenarios
            dist = norm(scenarios(i,:) - scenarios(j,:));
            distance_matrix(i,j) = dist;
            distance_matrix(j,i) = dist;
        end
    end
end

function [reduced_scenarios, probs, selected_indices] = ...
         fast_forward_selection(scenarios, probabilities, target_num, distance_matrix)
    
    num_scenarios = size(scenarios, 1);
    remaining_indices = 1:num_scenarios;
    selected_indices = [];
    
    % Step 1: Select first scenario - the one with minimum distance to all others
    total_distances = sum(distance_matrix, 2);
    [~, first_idx] = min(total_distances);
    
    selected_indices = first_idx;
    remaining_indices(remaining_indices == first_idx) = [];
    
    % Step 2: Iteratively add scenarios that are farthest from current selection
    while length(selected_indices) < target_num
        min_distances_to_selected = zeros(length(remaining_indices), 1);
        
        for i = 1:length(remaining_indices)
            idx = remaining_indices(i);
            min_distances_to_selected(i) = min(distance_matrix(idx, selected_indices));
        end
        
        [~, max_idx] = max(min_distances_to_selected);
        new_selected = remaining_indices(max_idx);
        
        selected_indices = [selected_indices, new_selected];
        remaining_indices(max_idx) = [];
    end
    
    % Calculate probabilities based on Voronoi cells
    reduced_scenarios = scenarios(selected_indices, :);
    probs = assign_probabilities(scenarios, probabilities, selected_indices, distance_matrix);
end

function [reduced_scenarios, probs, selected_indices] = ...
         backward_reduction(scenarios, probabilities, target_num, distance_matrix)
    
    num_scenarios = size(scenarios, 1);
    selected_indices = 1:num_scenarios;
    
    % Iteratively remove scenarios that contribute least to diversity
    while length(selected_indices) > target_num
        contribution = zeros(length(selected_indices), 1);
        
        for i = 1:length(selected_indices)
            temp_indices = selected_indices;
            temp_indices(i) = [];
            
            % Calculate minimum distance from removed scenario to remaining ones
            if ~isempty(temp_indices)
                contribution(i) = min(distance_matrix(selected_indices(i), temp_indices));
            else
                contribution(i) = 0;
            end
        end
        
        % Remove scenario with smallest contribution (closest to others)
        [~, min_idx] = min(contribution);
        selected_indices(min_idx) = [];
    end
    
    reduced_scenarios = scenarios(selected_indices, :);
    probs = assign_probabilities(scenarios, probabilities, selected_indices, distance_matrix);
end

function probs = assign_probabilities(scenarios, original_probs, selected_indices, distance_matrix)
    % Assign probabilities based on closest selected scenario (Voronoi cells)
    num_selected = length(selected_indices);
    probs = zeros(num_selected, 1);
    
    for i = 1:size(scenarios, 1)
        [~, closest_idx] = min(distance_matrix(i, selected_indices));
        probs(closest_idx) = probs(closest_idx) + original_probs(i);
    end
end

function quality = evaluate_reduction_quality(original_scenarios, original_probs, ...
                                            reduced_scenarios, reduced_probs, distance_matrix)
    % Calculate Kantorovich distance approximation
    num_original = size(original_scenarios, 1);
    num_reduced = size(reduced_scenarios, 1);
    
    kantorovich = 0;
    total_distance = 0;
    max_distance = 0;
    
    for i = 1:num_original
        min_dist = inf;
        for j = 1:num_reduced
            dist = norm(original_scenarios(i,:) - reduced_scenarios(j,:));
            min_dist = min(min_dist, dist);
        end
        kantorovich = kantorovich + original_probs(i) * min_dist;
        total_distance = total_distance + min_dist;
        max_distance = max(max_distance, min_dist);
    end
    
    % Calculate variance preservation
    original_var = trace(cov(original_scenarios));
    reduced_var = trace(weighted_cov(reduced_scenarios, reduced_probs));
    
    quality.kantorovich = kantorovich;
    quality.mean_distance = total_distance / num_original;
    quality.max_distance = max_distance;
    quality.variance_preserved = reduced_var / original_var;
end

function wcov = weighted_cov(data, weights)
    % Calculate weighted covariance matrix
    weights = weights / sum(weights);
    weighted_mean = weights' * data;
    centered_data = data - weighted_mean;
    wcov = centered_data' * (centered_data .* weights);
end

function wvar = weighted_variance(data, weights)
    % Calculate weighted variance for each time period
    num_time_periods = size(data, 2);
    wvar = zeros(1, num_time_periods);
    
    for t = 1:num_time_periods
        mean_val = weights' * data(:, t);
        wvar(t) = sum(weights .* (data(:, t) - mean_val).^2);
    end
end