# dualencoder

fMRI (X) → X_encoder → X_lat ╲
                              ╲ CLIP_corr → Diffusion/GAN → Y_result
Shapes (Y) → Y_encoder → Y_lat ╱           ╱
                               ╱          ╱
                         X_test ────────╱

% Encoding Phase
X_lat = fMRI_Encoder(X)           % fMRI → latent space
Y_lat = Shape_Encoder(Y)          % Shapes → latent space

% Correlation Learning  
CLIP_corr = CLIP_Correlation(X_lat, Y_lat)

% Decoding Phase
Y_result_diff = Diffusion(CLIP_corr, X_test)
Y_result_gan = GAN(CLIP_corr, X_test)

% Evaluation
accuracy_diff = Evaluate(Y_result_diff, Y_test)
accuracy_gan = Evaluate(Y_result_gan, Y_test)