% =========================================================================
% CNN FROM SCRATCH IN OCTAVE: Complete 0-9 Handcoded 9x9 Dataset & Training
% =========================================================================

clear; clc;

fprintf('--- 1. Initializing Complete 0-9 Handcoded 9x9 Dataset ---\n');

% --- DIGIT 0 ---
img_0 = zeros(9, 9);
img_0(2:8, 2) = 1; img_0(2:8, 8) = 1; 
img_0(2, 3:7) = 1; img_0(8, 3:7) = 1; 

% --- DIGIT 1 ---
img_1 = zeros(9, 9);
img_1(2:8, 5) = 1;                   
img_1(3, 4) = 1;                     

% --- DIGIT 2 ---
img_2 = zeros(9, 9);
img_2(2, 3:7) = 1;                   
img_2(3:4, 8) = 1;                   
img_2(5, 3:7) = 1;                   
img_2(6:7, 2) = 1;                   
img_2(8, 2:8) = 1;                   

% --- DIGIT 3 ---
img_3 = zeros(9, 9);
img_3(2, 3:7) = 1;                   
img_3(3:4, 8) = 1;                   
img_3(5, 4:7) = 1;                   
img_3(6:7, 8) = 1;                   
img_3(8, 3:7) = 1;                   

% --- DIGIT 4 ---
img_4 = zeros(9, 9);
img_4(2:6, 2) = 1;                   
img_4(5, 2:8) = 1;                   
img_4(2:8, 8) = 1;                   

% --- DIGIT 5 ---
img_5 = zeros(9, 9);
img_5(2, 2:8) = 1;                   
img_5(3:4, 2) = 1;                   
img_5(5, 2:7) = 1;                   
img_5(6:7, 8) = 1;                   
img_5(8, 2:7) = 1;                   

% --- DIGIT 6 ---
img_6 = zeros(9, 9);
img_6(2, 3:7) = 1;                   
img_6(3:8, 2) = 1;                   
img_6(5, 3:7) = 1;                   
img_6(6:7, 8) = 1;                   
img_6(8, 3:7) = 1;                   

% --- DIGIT 7 ---
img_7 = zeros(9, 9);
img_7(2, 2:8) = 1;                   
img_7(3:8, 7) = 1;                   

% --- DIGIT 8 ---
img_8 = zeros(9, 9);
img_8(2:8, 2) = 1; img_8(2:8, 8) = 1; 
img_8(2, 3:7) = 1;                   
img_8(5, 3:7) = 1;                   
img_8(8, 3:7) = 1;                   

% --- DIGIT 9 ---
img_9 = zeros(9, 9);
img_9(2:5, 2) = 1; img_9(2:5, 8) = 1; 
img_9(2, 3:7) = 1; img_9(5, 3:7) = 1; 
img_9(6:8, 8) = 1;                   
img_9(8, 3:7) = 1;                   

% Store all digits in a dataset cell array (Labels 1 to 10 for indexing convenience, representing digits 0-9)
dataset = { ...
    struct('matrix', img_0, 'label_idx', 1, 'name', '0'), ...
    struct('matrix', img_1, 'label_idx', 2, 'name', '1'), ...
    struct('matrix', img_2, 'label_idx', 3, 'name', '2'), ...
    struct('matrix', img_3, 'label_idx', 4, 'name', '3'), ...
    struct('matrix', img_4, 'label_idx', 5, 'name', '4'), ...
    struct('matrix', img_5, 'label_idx', 6, 'name', '5'), ...
    struct('matrix', img_6, 'label_idx', 7, 'name', '6'), ...
    struct('matrix', img_7, 'label_idx', 8, 'name', '7'), ...
    struct('matrix', img_8, 'label_idx', 9, 'name', '8'), ...
    struct('matrix', img_9, 'label_idx', 10, 'name', '9')  ...
};

fprintf('Dataset successfully loaded with 10 digit templates.\n\n');


fprintf('--- 2. Initializing Weights & Hyperparameters ---\n');

% Random 3x3 convolutional kernel
kernel = randn(3, 3) * 0.1;

% Dense layer weights: 7x7 flattened feature map = 49 inputs -> 10 output classes
W_dense = randn(10, 49) * 0.1;  % Shape: 10 x 49
b_dense = zeros(10, 1);         % Shape: 10 x 1

learning_rate = 0.05;
epochs = 1000;

fprintf('Initialized random 3x3 kernel and 10x49 dense classification layer.\n');
fprintf('Starting multi-class training for %d epochs...\n\n', epochs);


% =========================================================================
% --- 3. TRAINING LOOP ---
% =========================================================================

for epoch = 1:epochs
    total_loss = 0;
    
    for d = 1:length(dataset)
        input_image = dataset{d}.matrix;
        true_idx = dataset{d}.label_idx;
        
        % One-hot encoding for the true label (10x1 vector)
        y_true = zeros(10, 1);
        y_true(true_idx) = 1;
        
        % -----------------------------------------------------------------
        % FORWARD PASS
        % -----------------------------------------------------------------
        
        % A. Convolution (Valid Padding, Stride 1: 9x9 -> 7x7)
        out_height = 7; 
        out_width = 7;
        feature_map = zeros(out_height, out_width);
        
        for i = 1:out_height
            for j = 1:out_width
                patch = input_image(i:i+2, j:j+2);
                feature_map(i, j) = sum(sum(patch .* kernel));
            end
        end
        
        % ReLU Activation
        feature_map_relu = max(0, feature_map);
        
        % B. Flatten Layer (7x7 -> 49x1 Vector)
        flattened = reshape(feature_map_relu, [], 1);
        
        % C. Fully Connected (Dense) Layer + Softmax Activation
        z = W_dense * flattened + b_dense;              % Raw logits (10x1)
        exp_z = exp(z - max(z));                        % Numerical stability shift
        y_pred = exp_z / sum(exp_z);                    % Softmax probabilities (10x1)
        
        % D. Categorical Cross-Entropy Loss
        loss = -sum(y_true .* log(y_pred + 1e-8));
        total_loss += loss;
        
        % -----------------------------------------------------------------
        % BACKWARD PASS (Backpropagation)
        % -----------------------------------------------------------------
        
        % 1. Derivative of Loss w.r.t logits z (Softmax + CrossEntropy derivative is simply y_pred - y_true)
        dz = y_pred - y_true;                            % Shape: 10x1
        
        % 2. Dense Layer Gradients
        dL_dW_dense = dz * flattened';                   % Shape: 10x49
        dL_db_dense = dz;                                % Shape: 10x1
        
        % 3. Propagate backward through Flatten layer to 7x7 grid
        dL_dflattened = W_dense' * dz;                   % Shape: 49x1
        dL_dfeature_map = reshape(dL_dflattened, 7, 7);
        
        % Apply ReLU derivative (gradient passes only where feature_map > 0)
        dL_dfeature_map = dL_dfeature_map .* (feature_map > 0);
        
        % 4. Convolution Kernel Gradients
        dL_dkernel = zeros(3, 3);
        for i = 1:7
            for j = 1:7
                patch = input_image(i:i+2, j:j+2);
                dL_dkernel += patch * dL_dfeature_map(i, j);
            end
        end
        
        % -----------------------------------------------------------------
        % GRADIENT DESCENT UPDATE
        % -----------------------------------------------------------------
        kernel -= learning_rate * dL_dkernel;
        W_dense -= learning_rate * dL_dW_dense;
        b_dense -= learning_rate * dL_db_dense;
    end
    
    % Print training status every 1,000 epochs
    if mod(epoch, 200) == 0
        fprintf('Epoch %4d/%d | Average Loss: %.4f\n', epoch, epochs, total_loss / length(dataset));
    end
end


% =========================================================================
% --- 4. EVALUATION / TESTING ---
% =========================================================================

fprintf('\n--- Training Complete! Evaluating Model Predictions ---\n');
correct_predictions = 0;

for d = 1:length(dataset)
    input_image = dataset{d}.matrix;
    true_idx = dataset{d}.label_idx;
    digit_name = dataset{d}.name;
    
    % Forward pass test
    feature_map = zeros(7, 7);
    for i = 1:7
        for j = 1:7
            patch = input_image(i:i+2, j:j+2);
            feature_map(i, j) = sum(sum(patch .* kernel));
        end
    end
    feature_map_relu = max(0, feature_map);
    flattened = reshape(feature_map_relu, [], 1);
    z = W_dense * flattened + b_dense;
    exp_z = exp(z - max(z));
    y_pred = exp_z / sum(exp_z);
    
    [~, predicted_idx] = max(y_pred);
    predicted_digit = predicted_idx - 1; % Convert index back to digit (0-9)
    
    is_correct = (predicted_idx == true_idx);
    if is_correct
        correct_predictions += 1;
    end
    
    fprintf('True Digit: %s | Predicted: %s | Confidence: %.2f%% | Match: %s\n', ...
        digit_name, num2str(predicted_digit), y_pred(predicted_idx) * 100, mat2str(is_correct));
end

fprintf('\nFinal Accuracy: %.1f%%\n', (correct_predictions / length(dataset)) * 100);
