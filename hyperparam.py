optimal_hyperparams = {
    'fmri_encoder': {
        'hidden_dims': [2048, 1024],
        'dropout': 0.1,
        'activation': 'SiLU'
    },
    'shape_encoder': {
        'conv_channels': [32, 64, 128],
        'fc_dims': [512],
        'dropout': 0.05
    },
    'correlation': {
        'temperature': 0.07,
        'correlation_dim': 512
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs_phase1': 100,
        'epochs_phase2': 50,
        'epochs_phase3': 100
    }
}