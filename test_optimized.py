"""
Test script for the optimized suicide detection model.
Creates a small synthetic dataset to verify the code runs without errors.
"""

import pandas as pd
import numpy as np
import sys
import time

# Test data creation
def create_test_data(n_samples=100):
    """Create synthetic test data
    
    Args:
        n_samples: Number of samples to generate (default 100)
        
    Note: This creates a balanced 50/50 split for basic validation.
    For production testing, use real data with natural class imbalance.
    """
    np.random.seed(42)
    
    # Sample texts (mix of risk and non-risk phrases)
    risk_phrases = [
        "me quiero morir", "no puedo más", "estoy muy triste",
        "no veo futuro", "quiero parar", "me siento solo"
    ]
    
    normal_phrases = [
        "hoy tuve un buen día", "estoy contento con mi trabajo",
        "me gusta pasar tiempo con amigos", "el clima está agradable",
        "estoy aprendiendo cosas nuevas", "disfruto de mis hobbies"
    ]
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        if i % 2 == 0:  # Risk case
            text = np.random.choice(risk_phrases)
            label = 1
        else:  # Normal case
            text = np.random.choice(normal_phrases)
            label = 0
        
        # Add some variation
        text = text + f" y además {np.random.choice(['estoy', 'me siento', 'pienso que'])} "
        text = text + np.random.choice(['bien', 'mal', 'regular', 'cansado', 'tranquilo'])
        
        texts.append(text)
        labels.append(label)
    
    df = pd.DataFrame({
        'body_anonimizado': texts,
        'suicidi_ideacion_label': labels
    })
    
    return df

def test_optimized_code():
    """Test the optimized code with synthetic data"""
    print("="*70)
    print("TESTING OPTIMIZED SUICIDE DETECTION CODE")
    print("="*70)
    
    # Create test data
    print("\n📊 Creating synthetic test data...")
    test_df = create_test_data(n_samples=100)
    print(f"✅ Created {len(test_df)} samples")
    print(f"   - Class distribution: {test_df['suicidi_ideacion_label'].value_counts().to_dict()}")
    
    # Import the optimized module
    print("\n📦 Importing optimized module...")
    try:
        from optimized_suicide_detection import main_pipeline
        print("✅ Module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import module: {e}")
        return False
    
    # Run the pipeline
    print("\n🚀 Running optimized pipeline...")
    start_time = time.time()
    
    try:
        # Inject test data into globals for the pipeline
        import optimized_suicide_detection
        optimized_suicide_detection.merged_df1 = test_df
        
        # Run pipeline
        main_pipeline(test_df)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed successfully in {elapsed:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_code()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ TESTS FAILED")
        print("="*70)
        sys.exit(1)
