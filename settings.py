import os

class Settings():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(file_dir, "Logs")
    interface_info_path = os.path.join(file_dir, "interface_info.json")
    
    device = "cpu"
    frame_rate = 16000
    buffer_sec = 15
    buffer_max = buffer_sec*frame_rate
    chunk_sec = 1
    wait_chunks_for_VAD = 1
    max_time_sec = 8*60*60 # 8 hours

    class Interface():
        window_size = "800x600"
        title = "Emotion recognition from speech and text"
        TER_init_text = "cette demo est super!"

    class Models():
        file_dir = os.path.dirname(os.path.abspath(__file__))
        # VAD_feat_type = "MFB" # "MFB"
        # VAD_feat_norm=False
        # Linguistic_feat = "roberta-large"
        # SER_feat_type = "wav2vec2-xlsr" # "MFB"
        # SER_feat_norm=False
        # SER_dataset = "IEMOCAP"
        SER_models = {
            "A_2C_C": {
                "title": "Speech -> Multi (En) -> CMU_MOSEI",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_CMU_MOSEI+IEMOCAP_wav2vec2-xlsr-voidful_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "A_2C_I": {
                "title": "Speech -> Multi (En) -> IEMOCAP",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_CMU_MOSEI+IEMOCAP_wav2vec2-xlsr-voidful_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 1,
                    "dataset": "IEMOCAP",
                },
            },
            "A_MOSEI": {
                "title": "Speech -> CMU_MOSEI (En)",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "CMU_MOSEI_wav2vec2-xlsr-voidful_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "A_IEMOCAP": {
                "title": "Speech -> IEMOCAP (En)",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "IEMOCAP_wav2vec2-xlsr-voidful_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "IEMOCAP",
                },
            },
            "AL_IEMOCAP": {
                "title": "Speech +> ASR -> IEMOCAP (En)",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "roberta-large",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "AL_trs_cat", "IEMOCAP_wav2vec2-xlsr-voidful+roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "IEMOCAP",
                },
            },
            "AL_MOSEI": {
                "title": "Speech +> ASR -> CMU_MOSEI (En)",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "roberta-large",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "AL_trs_cat", "CMU_MOSEI_wav2vec2-xlsr-voidful+roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "AL_2C_C": {
                "title": "Speech +> ASR -> Multi (En) -> CMU_MOSEI",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "roberta-large",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "AL_trs_cat", "multi_CMU_MOSEI+IEMOCAP_wav2vec2-xlsr-voidful+roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "AL_2C_I": {
                "title": "Speech +> ASR -> Multi (En) -> IEMOCAP",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "roberta-large",
                "feat_norm_text": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "AL_trs_cat", "multi_CMU_MOSEI+IEMOCAP_wav2vec2-xlsr-voidful+roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 1,
                    "dataset": "IEMOCAP",
                },
            },
            "A_4C_G": {
                "title": "Speech -> Multi (Fr) -> GEMEP",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "fr-FR",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_GEMEP_RAVDESS_CaFE_EmoDB_wav2vec2-xlsr_GRU-1-64_lr-0.0001", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "GEMEP",
                },
            },
            "A_4C_R": {
                "title": "Speech -> Multi (Fr) -> RAVDESS",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "fr-FR",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_GEMEP_RAVDESS_CaFE_EmoDB_wav2vec2-xlsr_GRU-1-64_lr-0.0001", "model.pth"),
                "classifier": {
                    "index": 1,
                    "dataset": "RAVDESS",
                },
            },
            "A_4C_C": {
                "title": "Speech -> Multi (Fr) -> CaFE",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "fr-FR",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_GEMEP_RAVDESS_CaFE_EmoDB_wav2vec2-xlsr_GRU-1-64_lr-0.0001", "model.pth"),
                "classifier": {
                    "index": 2,
                    "dataset": "CaFE",
                },
            },
            "A_4C_E": {
                "title": "Speech -> Multi (Fr) -> EmoDB",
                "feat_type": "wav2vec2-xlsr",
                "feat_norm": False,
                "feat_type_text": "",
                "feat_norm_text": False,
                "lang": "fr-FR",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "multi_GEMEP_RAVDESS_CaFE_EmoDB_wav2vec2-xlsr_GRU-1-64_lr-0.0001", "model.pth"),
                "classifier": {
                    "index": 3,
                    "dataset": "EmoDB",
                },
            },
            "AL_IEMOCAP_Allociné": {
                "title": "S -> IEMOCAP (A), ASR -> Allociné (V)",
                "feat_type": "logfbank",
                "feat_norm": True,
                "feat_type_text": "tfidf",
                "feat_norm_text": False,
                "lang": "fr-Fr",
                "model_path": os.path.join(file_dir, "Models", "SER", "A", "sklearn", "audio_mfb_mlp_arousal.sav"),
                "model_path_v": os.path.join(file_dir, "Models", "SER", "A", "sklearn", "audio_mfb_mlp_valence.sav"),
                "model_path_trs": os.path.join(file_dir, "Models", "SER", "L", "sklearn", "text_tdidf_mlp.sav"),
                "vectorizer_path": os.path.join(file_dir, "Models", "SER", "L", "sklearn", "allociné.tsv"),
                "classifier": {
                    "index": 0,
                    "dataset": "IEMOCAP-Allociné",
                },
            },
        }
        TER_models = {
            "L_IEMOCAP": {
                "title": "Human transcription -> IEMOCAP (En)",
                "feat_type": "roberta-large",
                "feat_norm": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "L", "IEMOCAP_roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "IEMOCAP",
                },
            },
            "L_IEMOCAP_trs": {
                "title": "ASR transcription -> IEMOCAP (En)",
                "feat_type": "roberta-large",
                "feat_norm": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "L_trs", "IEMOCAP_roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "IEMOCAP",
                },
            },
            "L_MOSEI_trs": {
                "title": "ASR transcription -> CMU_MOSEI (En)",
                "feat_type": "roberta-large",
                "feat_norm": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "L_trs", "CMU_MOSEI_roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "L_2C_trs_C": {
                "title": "ASR transcription -> Multi (En) -> CMU_MOSEI",
                "feat_type": "roberta-large",
                "feat_norm": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "L_trs", "multi_CMU_MOSEI+IEMOCAP_roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 0,
                    "dataset": "CMU_MOSEI",
                },
            },
            "L_2C_trs_I": {
                "title": "ASR transcription -> Multi (En) -> IEMOCAP",
                "feat_type": "roberta-large",
                "feat_norm": False,
                "lang": "en-US",
                "model_path": os.path.join(file_dir, "Models", "SER", "L_trs", "multi_CMU_MOSEI+IEMOCAP_roberta-large_GRU_bimodal-1-64_lr-0.0001_bs-1_ga-100", "model.pth"),
                "classifier": {
                    "index": 1,
                    "dataset": "IEMOCAP",
                },
            },
            "L_allociné": {
                "title": "Human transcription -> Allociné (Fr)",
                "feat_type": "tfidf",
                "feat_norm": False,
                "lang": "fr-Fr",
                "model_path": os.path.join(file_dir, "Models", "SER", "L", "sklearn", "text_tdidf_mlp.sav"),
                "vectorizer_path": os.path.join(file_dir, "Models", "SER", "L", "sklearn", "allociné.tsv"),
                "classifier": {
                    "index": 0,
                    "dataset": "Allociné",
                },
            },
        }
        VAD_models = {
            "MFB_Lin": {
                "title": "MFB -> Lin",
                "feat_type": "MFB",
                "feat_norm": True,
                "model_path": os.path.join(file_dir, "Models", "VAD", "MFB_Lin", "model.pth"),
                "post_process":{
                    "hys_top":      0.5,
                    "hys_bottom":   -0.4,
                    "cutWin_sec":   0.1,
                    "mergeWin_sec": 0.7,
                }
            },
            "MFB_GRU": {
                "title": "MFB -> GRU",
                "feat_type": "MFB",
                "feat_norm": True,
                "model_path": os.path.join(file_dir, "Models", "VAD", "MFB_GRU", "model.pth"),
                "post_process":{
                    "hys_top":      0.5,
                    "hys_bottom":   0.0,
                    "cutWin_sec":   0.1,
                    "mergeWin_sec": 0.7,
                }
            },
            "W2V2_GRU": {
                "title": "W2V2 -> GRU",
                "feat_type": "wav2vec2-FR-2.6K-base",
                "feat_norm": False,
                "model_path": os.path.join(file_dir, "Models", "VAD", "W2V2_GRU", "model.pth"),
                "post_process":{
                    "hys_top":      -0.2,
                    "hys_bottom":   -0.25,
                    "cutWin_sec":   0.1,
                    "mergeWin_sec": 0.7,
                }
            },
        }

        
        