class CorrelationBank:
    """Store and retrieve correlations for test phase"""
    def __init__(self):
        self.correlations = {}
    
    def store_correlation(self, sample_id, CLIP_corr):
        self.correlations[sample_id] = CLIP_corr.detach()
    
    def get_correlation(self, sample_id):
        return self.correlations.get(sample_id, None)