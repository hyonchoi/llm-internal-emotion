from emotion_tracer.config import EmotionTracerConfig

def test_default_emotions():
    cfg = EmotionTracerConfig()
    assert "happy" in cfg.emotions
    assert "sad" in cfg.emotions
    assert len(cfg.emotions) == 20

def test_resolve_layer_indices_default():
    cfg = EmotionTracerConfig()
    # With 32 layers, 2/3 depth = index 21
    indices = cfg.resolve_layer_indices(32)
    assert indices == [21]

def test_resolve_layer_indices_explicit():
    cfg = EmotionTracerConfig(layer_indices=[10, 20])
    indices = cfg.resolve_layer_indices(32)
    assert indices == [10, 20]

def test_default_device_is_cpu():
    cfg = EmotionTracerConfig()
    assert cfg.device == "cpu"
