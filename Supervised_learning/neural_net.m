clc;
clear;
close all;

### Input Features ###
num_features = input("Enter the number of input features: ");
fprintf("Number of neurons in input layer: %d\n", num_features);

### Hidden Layer Architecture ###
num_hidden_layers = input("Enter the number of hidden layers: ");
fprintf("Number of hidden layers: %d\n", num_hidden_layers);

### Setting Hidden Layer Architecture ###
hidden_neuron = zeros(1, num_hidden_layers);
for i = 1:1:num_hidden_layers
    hidden_neuron(i) = input(sprintf('Enter neuron in %d layer: ', i));
endfor

hidden_neuron

### Input Weights ###
input_weight = ones(1, num_features);

### Input and Output Set ### (XOR feature)
input_test_data_map = [0, 0; 0, 1; 1, 0; 1, 1];
output_test_data_map = [0; 1; 1; 0];

### Activation Functions & Derivatives ###
sigmoid = @(z) 1 ./ (1 + exp(-z));
tanh_act = @(z) tanh(z);
relu = @(z) max(0, z);

sigmoid_prime = @(a) a .* (1 - a);
relu_prime = @(z) double(z > 0);

### Constructing the Architecture Layer-by-layer ###
layer_template = struct('layer_id', 0, 'num_neurons', 0, 'weights', [], 'bias', [], 'activation', relu);

input_layer = repmat(layer_template, 1, 1);
input_layer.num_neurons = num_features;
input_layer.weights = input_weight;

output_layer = repmat(layer_template, 1, 1);
output_layer.num_neurons = 1; % XOR has 1 output
output_layer.activation = sigmoid;

%%% Initializing Hidden Layers Weights & Biases %%%
hidden_layers = repmat(layer_template, num_hidden_layers, 1);
prev_neurons = num_features;

for i = 1:num_hidden_layers
    hidden_layers(i).layer_id = i;
    hidden_layers(i).num_neurons = hidden_neuron(i);
    hidden_layers(i).weights = randn(hidden_neuron(i), prev_neurons);
    hidden_layers(i).bias = zeros(hidden_neuron(i), 1);
    hidden_layers(i).activation = relu;
    prev_neurons = hidden_neuron(i);
endfor

%%% Initializing Output Layer Weights %%%
output_layer.weights = randn(output_layer.num_neurons, prev_neurons);
output_layer.bias = zeros(output_layer.num_neurons, 1);

fprintf("\nNetwork Architecture Initialized Successfully!\n");

%%% Training Hyperparameters %%%
learning_rate = 0.1;
epochs = 10000;

X = input_test_data_map'; % Shape: 2 x 4
Y = output_test_data_map'; % Shape: 1 x 4

fprintf("Starting Training...\n");

for epoch = 1:epochs
    
    %% 1. FORWARD PROPAGATION %%
    layer_inputs = cell(num_hidden_layers + 1, 1);
    layer_outputs = cell(num_hidden_layers + 1, 1);
    
    current_activation = X;
    
    for i = 1:num_hidden_layers
        layer_inputs{i} = hidden_layers(i).weights * current_activation + hidden_layers(i).bias;
        current_activation = hidden_layers(i).activation(layer_inputs{i});
        layer_outputs{i} = current_activation;
    endfor
    
    out_idx = num_hidden_layers + 1;
    layer_inputs{out_idx} = output_layer.weights * current_activation + output_layer.bias;
    predicted_y = output_layer.activation(layer_inputs{out_idx});
    layer_outputs{out_idx} = predicted_y;
    
    %% 2. LOSS CALCULATION (Mean Squared Error) %%
    loss = mean(sum((predicted_y - Y).^2));
    
    if mod(epoch, 1000) == 0
        fprintf('Epoch %d/%d - Loss: %.6f\n', epoch, epochs, loss);
    endif
    
    %% 3. BACKPROPAGATION %%
    d_loss = (predicted_y - Y);
    delta = d_loss .* sigmoid_prime(predicted_y);
    
    num_samples = size(X, 2);
    prev_activation = layer_outputs{num_hidden_layers};
    
    dW_out = (delta * prev_activation') / num_samples;
    db_out = sum(delta, 2) / num_samples;
    
    dW = cell(num_hidden_layers, 1);
    db = cell(num_hidden_layers, 1);
    
    current_delta = delta;
    current_weights = output_layer.weights;
    
    for i = num_hidden_layers:-1:1
        next_delta = (current_weights' * current_delta) .* relu_prime(layer_inputs{i});
        
        if i == 1
            prev_act = X;
        else
            prev_act = layer_outputs{i-1};
        endif
        
        dW{i} = (next_delta * prev_act') / num_samples;
        db{i} = sum(next_delta, 2) / num_samples;
        
        current_delta = next_delta;
        current_weights = hidden_layers(i).weights;
    endfor
    
    %% 4. GRADIENT DESCENT WEIGHT UPDATE %%
    output_layer.weights = output_layer.weights - learning_rate * dW_out;
    output_layer.bias = output_layer.bias - learning_rate * db_out;
    
    for i = 1:num_hidden_layers
        hidden_layers(i).weights = hidden_layers(i).weights - learning_rate * dW{i};
        hidden_layers(i).bias = hidden_layers(i).bias - learning_rate * db{i};
    endfor
    
endfor

fprintf("\nTraining Complete! Final XOR Predictions:\n");
disp(predicted_y');
