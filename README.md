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


============================================================
SUMMARY PERBANDINGAN DATASET
============================================================
detailed_image_analysis.py


CRELL:
  fmriTrn: (576, 3092) (float64)
  fmriTest: (64, 3092) (float64)
  stimTrn: (576, 784) (uint8)
  stimTest: (64, 784) (uint8)
  labelTrn: (576, 1) (uint8)
  labelTest: (64, 1) (uint8)

DIGIT69_28X28:
  fmriTest: (10, 3092) (float64)
  fmriTrn: (90, 3092) (float64)
  labelTest: (10, 1) (uint8)
  labelTrn: (90, 1) (uint8)
  stimTest: (10, 784) (uint8)
  stimTrn: (90, 784) (uint8)

MINDBIGDATA:
  fmriTrn: (1080, 3092) (float64)
  fmriTest: (120, 3092) (float64)
  stimTrn: (1080, 784) (uint8)
  stimTest: (120, 784) (uint8)
  labelTrn: (1080, 1) (uint8)
  labelTest: (120, 1) (uint8)