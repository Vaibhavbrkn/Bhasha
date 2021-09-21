from dataclasses import dataclass, field
from typing import List, Any, Optional, Tuple
from transformers import (BertConfig, AlbertConfig, CamembertConfig,
                          BartConfig, BigBirdConfig,
                          BigBirdPegasusConfig, BlenderbotConfig,
                          BlenderbotSmallConfig, CamembertConfig,
                          CanineConfig, ConvBertConfig,
                          CTRLConfig, DebertaConfig,
                          DebertaV2Config, DistilBertConfig,
                          ElectraConfig, FlaubertConfig,
                          FunnelConfig, GPT2Config,
                          GPTNeoConfig, HubertConfig,
                          IBertConfig, LayoutLMConfig,
                          LayoutLMv2Config, LEDConfig,
                          LongformerConfig, LukeConfig,
                          LxmertConfig, M2M100Config,
                          MarianConfig, MBartConfig,
                          MegatronBertConfig, MobileBertConfig,
                          MPNetConfig, MT5Config,
                          OpenAIGPTConfig, PegasusConfig,
                          ProphetNetConfig, RagConfig,
                          ReformerConfig, RemBertConfig,
                          RobertaConfig, RoFormerConfig,
                          SplinterConfig, SqueezeBertConfig,
                          T5Config, TapasConfig,
                          TransfoXLConfig, XLMConfig,
                          XLMProphetNetConfig, XLMRobertaConfig,
                          XLNetConfig)


@dataclass
class TrainParams:
    batch_size: int = 1
    val_size: float = .2
    max_len: int = 512
    learning_rate: float = 2e-5
    optimizer: str = 'Adam'
    # Learning Rate Schedule
    lr_schedule: str = "ReduceLROnPlateau"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_base: float = 0.01
    lr_max: float = 0.1
    lr_mode: str = 'min'
    lr_patience: int = 5
    lr_factor: float = 0.1
    lr_threshold: float = 1e-4
    lr_milestones: List[int] = field(default_factory=lambda: [5, 10])
    # Optimizer
    betas: Tuple = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False
    centered: bool = False
    alpha: float = 0.99
    rho: float = 0.9
    lr_decay: float = 0


@dataclass
class ConfigParams:

    @dataclass
    class Bert:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = "absolute"
        use_cache = bool = True
        classifier_dropout: Optional[float] = None

    @dataclass
    class Albert:
        vocab_size: int = 30000
        embedding_size: Optional[int] = 128
        hidden_size: int = 4096
        num_hidden_layers: int = 12
        num_hidden_groups: Optional[int] = 1
        num_attention_heads: int = 64
        intermediate_size: int = 16384
        inner_group_num: Optional[int] = 1
        hidden_act: str = 'gelu_new'
        hidden_dropout_prob: float = 0
        attention_probs_dropout_prob: float = 0
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        classifier_dropout_prob: float = 0.1
        position_embedding_type: str = "absolute"

    @dataclass
    class Bart:
        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        scale_embedding: bool = False
        use_cache: bool = True
        num_labels: int = 3
        forced_eos_token_id: int = 2

    @dataclass
    class BigBird:
        vocab_size: int = 50358
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu_new'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_probs: float = 0.1
        max_position_embeddings: int = 4096
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        use_cache: bool = True
        attention_type: str = 'block_sparse'
        use_bias: bool = True
        rescale_embeddings: bool = True
        block_size: int = 64
        num_random_blocks: int = 3
        gradient_checkpointing: bool = False

    @dataclass
    class BigBirdPegasus:
        vocab_size: int = 96103
        d_model: int = 1024
        encoder_layers: int = 16
        decoder_layers: int = 16
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu_new'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 4096
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        use_cache: bool = True
        attention_type: str = "block_sparse"
        use_bias: bool = False
        block_size: int = 64
        num_random_blocks: int = 3
        scale_embeddings: bool = True
        gradient_checkpointing: bool = False

    @dataclass
    class Blenderbot:
        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 16
        decoder_layers: int = 16
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 128
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        use_cache: bool = True
        forced_eos_token_id: int = 2
        scale_embeddings: bool = False
        gradient_checkpointing: bool = False

    @dataclass
    class BlenderbotSmall:
        vocab_size: int = 50265
        d_model: int = 512
        encoder_layers: int = 8
        decoder_layers: int = 8
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 2048
        encoder_ffn_dim: int = 2048
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 512
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        use_cache: bool = True
        forced_eos_token_id: int = 2
        scale_embeddings: bool = False
        gradient_checkpointing: bool = False

    @dataclass
    class Camembert:

        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = "absolute"
        use_cache = bool = True
        classifier_dropout: Optional[float] = None

    @dataclass
    class Canine:
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 16384
        type_vocab_size: int = 16
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        downsampling_rate: int = 4
        upsampling_kernel_size: int = 4
        num_hash_functions: int = 8
        num_hash_buckets: int = 16384
        local_transformer_stride: int = 128

    @dataclass
    class ConvBert:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        head_ratio: int = 2
        num_groups: int = 1
        conv_kernel_size: int = 9

    @dataclass
    class CTRL:
        vocab_size: int = 246534
        n_positions: int = 256
        n_ctx: int = 256
        n_embed: int = 1280
        diff: int = 8192
        n_layer: int = 48
        n_head: int = 16
        resid_pdrop: float = 0.1
        embd_pdrop: float = 0.1
        hidden_dropout_prob: float = 0.1
        attn_pdrop: float = 0.1
        initializer_range: float = 0.02
        layer_norm_epsilon: float = 1e-6
        use_cache: bool = True

    @dataclass
    class Deberta:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        relative_attention: bool = False
        max_relative_positions: int = 1
        pad_token_id: int = 0
        max_relative_positions: bool = True

    @dataclass
    class DebertaV2:
        vocab_size: int = 128100
        hidden_size: int = 1536
        num_hidden_layers: int = 24
        num_attention_heads: int = 24
        intermediate_size: int = 6144
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 0
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        relative_attention: bool = True
        max_relative_positions: int = -1
        pad_token_id: int = 0
        max_relative_positions: bool = False

    @dataclass
    class DistilBert:
        vocab_size: int = 30522
        max_position_embeddings: int = 512
        sinusoidal_pos_embds: bool = False
        n_layers: int = 6
        n_heads: int = 12
        dim: int = 768
        hidden_dim: int = 3072
        dropout: float = 0.1
        attention_dropout: float = 0.1
        activtion: str = 'gelu'
        qa_dropout: float = 0.1
        seq_classif_dropout = float = 0.2

    @dataclass
    class DPR:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = 'absolute'
        projection_dim: int = 0

    @dataclass
    class Electra:
        vocab_size: int = 30522
        embedding_size: int = 128
        hidden_size: int = 256
        num_hidden_layers: int = 12
        num_attention_heads: int = 4
        intermediate_size: int = 1024
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        summary_type: str = 'first'
        summary_use_proj: bool = True
        summary_activation: str = 'gelu'
        summary_last_dropout: float = 0.0
        position_embedding_type: str = 'absolute'

    @dataclass
    class Flaubert:
        pre_norm: bool = False
        layerdrop: float = 0.0
        vocab_size: int = 30145
        emb_dim: int = 2048
        n_layer: int = 12
        n_head: int = 16
        dropout: float = 0.1
        attention_dropout: float = 0.1
        gelu_activation: bool = True
        sinusoidal_embeddings: bool = False
        casual: bool = False
        asm: bool = False
        n_langs: int = 1
        use_lang_emb: bool = True
        max_position_embeddings: int = 512
        embed_init_std: float = 2048**-0.5
        init_std: int = 50257
        layer_norm_eps: float = 1e-12
        bos_index: int = 0
        eos_index: int = 1
        pad_index: int = 2
        unk_index: int = 3
        mask_index: int = 5
        is_encoder: bool = True
        summary_type: str = 'first'
        summary_use_proj: bool = True
        summary_proj_to_labels: bool = True
        summary_first_dropout: float = 0.1
        start_n_top: int = 5
        end_n_top: int = 5
        mask_token_id: int = 0
        lang_id: int = 1

    @dataclass
    # Exception
    class FSMT:
        langs: List[str]
        src_vocab_size: int = 42024
        tgt_vocab_size: int = 42024
        d_model = int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'relu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        scale_embedding: bool = True
        bos_token_id: int = 0
        pad_token_id: int = 1
        eos_token_id: int = 2
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        is_encoder_decoder: bool = True
        tie_word_embeddings: bool = False
        num_beams: int = 5
        lenght_penalty: float = 1.
        early_stopping: bool = False
        use_cache: bool = True
        forced_eos_token_id: int = 2

    @dataclass
    class Funnel:
        vocab_size: int = 30522
        block_size: List[int] = field(default_factory=lambda: [4, 4, 4])
        num_decoder_layers: int = 2
        d_model: int = 768
        n_head: int = 12
        d_head: int = 64
        d_inner: int = 3072
        hidden_act: str = 'gelu_new'
        hidden_dropout: float = 0.1
        attention_dropout: float = 0.1
        activation_dropout: float = 0.0
        max_position_embeddings: int = 512
        type_vocab_size: int = 3
        initializer_range: float = 0.1
        layer_norm_eps: float = 1e-9
        pooling_type: str = 'mean'
        attention_type: str = 'relative_shift'
        separate_cls: bool = True
        truncate_seq: bool = False
        pool_q_only: bool = False

    @dataclass
    class GPT2:
        vocab_size: int = 50257
        n_positions: int = 1024
        n_ctx: int = 1024
        n_embed: int = 768
        n_layer: int = 12
        n_head: int = 12
        n_inner: int = None
        activation_function: str = 'gelu'
        resid_pdrop: float = 0.1
        embed_pdrop: float = 0.1
        attn_pdrop: float = 0.1
        layer_norm_epsilon: float = 1e-5
        initializer_range: float = 0.02
        summary_type: str = 'cls_index'
        summary_use_proj: bool = True
        summary_proj_to_labels: bool = True
        summary_first_dropout: float = 0.1
        scale_attn_weights: bool = True
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class GPTNeo:
        vocab_size: int = 50257
        hidden_size: int = 2048
        num_layers: int = 24
        num_heads: int = 16
        intermediate_size: int = 8192
        activation_function: str = 'gelu_new'
        embed_dropout: float = 0.0
        attention_dropout: float = 0.0
        max_position_embeddings: int = 2048
        type_vocab_size: int = 2
        layer_norm_epsilon: float = 1e-5
        initializer_range: float = 0.02
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class Hubert:
        vocab_size: int = 32
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        initializer_range: float = 0.02
        layer_norms_eps: float = 1e-12
        feat_extract_norm: str = 'group'
        feat_extract_dropout: float = 0.0
        feat_extract_activation: str = 'gelu'
        conv_bias: bool = False
        num_conv_pos_embeddings: int = 128
        num_conv_pos_embedding_groups: int = 16
        do_stable_layer_norm: bool = False
        apply_spec_augment: bool = True
        mask_time_prob: float = 0.05
        mask_time_length: int = 10
        mask_feature_prob: float = 0.0
        mask_feature_length: int = 10
        ctc_loss_reduction: str = "sum"
        ctc_zero_infinity: bool = "false"
        use_weighted_layer_sum: bool = False
        classifier_proj_size: int = 256
        gradient_checkpointing: bool = False

    @dataclass
    class IBert:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        position_embedding_type: str = 'absolute'
        quant_mode: bool = False
        force_dequant: str = "none"

    @dataclass
    class LayoutLM:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        max_2d_position_embeddings: int = 1024

    @dataclass
    class LayoutLMv2:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        max_2d_position_embeddings: int = 1024
        max_rel_pos: int = 128
        rel_pos_bins: int = 32
        fast_qkv: bool = True
        max_rel_2d_pos: int = 256
        rel_2d_pos_bins: int = 64
        image_feature_pool_shape: List[int] = field(
            default_factory=lambda: [7, 7, 256])
        coordinate_size: int = 128
        shape_size: int = 128
        has_relative_attention_bias: bool = True
        has_spatial_attention_bias: bool = True
        has_visual_segment_embedding: bool = False

    @dataclass
    class LED:
        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_encoder_position_embeddings: int = 16384
        max_decoder_position_embeddings: int = 16384
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class Longformer:
        attention_window: int = 512

    @dataclass
    class Luke:
        vocab_size: int = 30522
        entity_vocab_size: int = 500000
        hidden_size: int = 768
        entity_emb_size: int = 256
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        use_entity_aware_attention: bool = True

    @dataclass
    class Lxmert:
        vocab_size: int = 30522
        hidden_size: int = 768
        r_layers: int = 5
        l_layers: int = 9
        x_layers: int = 5
        num_attention_heads: int = 5
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        visual_feat_dim: int = 2048
        visual_pos_dim: int = 4
        visual_loss_normalizer: float = 1/15
        num_qa_labels: int = 9500
        num_object_labels: int = 1600
        num_attr_labels: int = 400
        task_matched: bool = True
        task_mask_lm: bool = True
        task_obj_predict: bool = True
        task_qa: bool = True
        visual_obj_loss: bool = True
        visual_attr_loss: bool = True
        visual_feat_loss: bool = True
        output_attentions: bool = False
        output_hidden_states: bool = False

    @dataclass
    class M2M100:

        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class Marian:

        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        scale_embedding: bool = False
        use_cache: bool = True
        forced_eos_token_id: int = 0

    @dataclass
    class MBart:

        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        scale_embedding: bool = False
        use_cache: bool = True
        forced_eos_token_id: int = 0

    @dataclass
    class MegatronBert:

        vocab_size: int = 29056
        hidden_size: int = 1024
        num_hidden_layers: int = 24
        num_attention_heads: int = 16
        intermediate_size: int = 4096
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = "absolute"
        use_cache = bool = True

    @dataclass
    class MobileBert:

        vocab_size: int = 30522
        hidden_size: int = 512
        num_hidden_layers: int = 24
        num_attention_heads: int = 4
        intermediate_size: int = 512
        hidden_act: str = 'relu'
        hidden_dropout_prob: float = 0.0
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        pad_token_id: int = 0
        embedding_size: int = 128
        trigram_input: bool = True
        use_bottleneck: bool = True
        intra_bottleneck_size: int = 128
        use_bottleneck_attention: bool = False
        key_query_shared_bottleneck: bool = True
        num_feedforward_networks: int = 4
        normalization_type: str = 'no_norm'
        classifier_dropout: Optional[float] = None

    @dataclass
    class MPNet:

        vocab_size: int = 30527
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        relative_attention_num_buckets: int = 32

    @dataclass
    class MT5:
        vocab_size: int = 250112
        d_model: int = 512
        d_kv: int = 84
        d_ff: int = 1024
        num_layers: int = 8
        num_decoder_layers: int = 8
        num_heads: int = 6
        relative_attention_num_buckets: int = 32
        dropout_rate: float = 0.1
        layer_norm_eps: float = 1e-6
        initializer_factor: float = 1
        feed_forward_proj: str = 'gated-gelu'
        use_cache: bool = True

    @dataclass
    class OpenAIGPT:

        vocab_size: int = 40478
        n_positions: int = 512
        n_ctx: int = 512
        n_embed: int = 768
        n_layer: int = 12
        n_head: int = 12
        afn: str = 'gelu'
        resid_pdrop: float = 0.1
        embd_pdrop: float = 0.1
        attn_pdrop: float = 0.1
        layer_norm_epsilon: float = 1e-5
        initializer_range: float = 0.02
        predict_special_tokens: bool = True
        summary_type: str = 'cls_index'
        summary_use_proj: bool = True
        summary_proj_to_labels: bool = True
        summary_proj_to_labels: float = 0.1
        use_cache: bool = True

    @dataclass
    class Pegasus:

        vocab_size: int = 50265
        d_model: int = 1024
        encoder_layers: int = 12
        decoder_layers: int = 12
        encoder_attention_heads: int = 16
        decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        activation_function: str = 'gelu'
        dropout: float = 0.1
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        classifier_dropout: float = 0.0
        max_position_embeddings: int = 1024
        init_std: float = 0.02
        encoder_layerdrop: float = 0.0
        decoder_layerdrop: float = 0.0
        gradient_checkpointing: bool = False
        scale_embedding: bool = False
        use_cache: bool = True
        forced_eos_token_id: int = 0

    @dataclass
    class ProphetNet:

        activation_dropout: float = 0.1
        activation_function: str = 'gelu'
        vocab_size: int = 30522
        hidden_size: int = 1024
        num_encoder_layers: int = 12
        num_decoder_layers: int = 12
        num_encoder_attention_heads: int = 16
        num_decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        dropout: float = 0.1
        attention_dropout: float = 0.1
        max_position_embeddings: int = 512
        init_std: float = 0.02
        add_cross_attention: bool = True
        is_encoder_decoder: bool = True
        ngram: int = 2
        num_buckets: int = 32
        relative_max_distance: int = 128
        disable_ngram_loss: bool = False
        eps: float = 0.0
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class Rag:
        title_sep: str = " / "
        doc_sep: str = " // "
        n_docs: int = 5
        max_combined_length: int = 300
        retrieval_vector_size: int = 768
        retrieval_batch_size: int = 8
        dataset: str = "wiki_dpr"
        dataset_split: str = "train"
        index_name: str = "compressed"
        use_dummy_dataset: bool = False
        label_smoothing: float = 0.0
        do_marginalize: bool = False
        reduce_loss: bool = False
        do_deduplication: bool = True
        exclude_bos_score: bool = False
        output_retrieved: bool = False
        use_cache: bool = True

    @dataclass
    class Reformer:
        attention_head_size: int = 64
        attn_layers: List[str] = field(default_factory=lambda: [
                                       "local", "lsh", "local", "lsh", "local", "lsh"])
        axial_pos_embds: bool = True
        axial_norm_std: float = 1.0
        axial_pos_shape: List[int] = field(default_factory=lambda: [64, 64])
        axial_pos_embds_dim: List[int] = field(
            default_factory=lambda: [64, 192])
        chunk_size_lm_head: int = 0
        eos_token_id: int = 2
        feed_forward_size: int = 512
        hidden_act: str = 'relu'
        hidden_dropout_prob: float = 0.05
        hidden_size: int = 256
        initializer_range: float = 0.02
        is_decoder: bool = False
        layer_norm_eps: float = 1e-12
        local_chunk_length: int = 64
        local_num_chunks_before: int = 1
        local_num_chunks_after: int = 0
        local_attention_probs_dropout_prob: float = 0.1
        lsh_attn_chunk_length: int = 64
        lsh_num_chunks_before: int = 1
        lsh_num_chunks_after: int = 0
        lsh_attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 4096
        num_attention_heads: int = 12
        num_hashes: int = 1
        pad_token_id: int = 0
        vocab_size: int = 320
        tie_word_embeddings: bool = False
        use_cache: bool = True

    @dataclass
    class RemBert:
        vocab_size: int = 250300
        hidden_size: int = 1152
        num_hidden_layers: int = 32
        num_attention_heads: int = 18
        input_embedding_size: int = 256
        output_embedding_size: int = 1664
        intermediate_size: int = 4608
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0
        attention_probs_dropout_prob: float = 0
        classifier_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        use_cache: bool = True
        gradient_checkpointing: bool = False

    @dataclass
    class Roberta:

        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = "absolute"
        use_cache = bool = True
        classifier_dropout: Optional[float] = None

    @dataclass
    class RoFormer:
        vocab_size: int = 50000
        embedding_size: int = 768
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 1536
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        use_cache = bool = True
        rotary_value: bool = False

    @dataclass
    class Splinter:

        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        use_cache = bool = True
        question_token_id: int = 104

    @dataclass
    class SqueezeBert:

        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        pad_token_id: int = 0
        embedding_size: int = 768
        q_groups: int = 4
        k_groups: int = 4
        v_groups: int = 4
        post_attention_groups: int = 1
        intermediate_groups: int = 4
        output_groups: int = 4

    @dataclass
    class T5:

        vocab_size: int = 32128
        d_model: int = 512
        d_kv: int = 64
        d_ff: int = 2048
        num_layers: int = 6
        num_decoder_layers: int = 6
        num_heads: int = 8
        relative_attention_num_buckets: int = 32
        dropout_rate: float = 0.1
        layer_norm_eps: float = 1e-6
        initializer_factor: float = 1
        feed_forward_proj: str = 'relu'
        use_cache: bool = True
        gradient_checkpointing: bool = False

    @dataclass
    class Tapas:
        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 1024
        type_vocab_sizes: List[int] = field(
            default_factory=lambda: [3, 256, 256, 2, 256, 256, 10])
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        positive_label_weight: float = 10.0
        num_aggregation_labels: int = 0
        aggregation_loss_weight: float = 1.0
        answer_loss_importance: float = 1.0
        use_normalized_answer_loss: bool = False
        temperature: float = 1.0
        aggregation_temperature: float = 1.0
        use_gumbel_for_cells: bool = False
        use_gumbel_for_aggregation: bool = False
        average_approximation_function: str = 'ratio'
        max_num_rows: int = 64
        max_num_columns: int = 32
        average_logits_per_cell: bool = False
        select_one_column: bool = True
        allow_empty_column_selection: bool = False
        init_cell_selection_weights_to_zero: bool = False
        reset_position_index_per_cell: bool = True
        disable_per_token_loss: bool = False

    @dataclass
    class TransfoXL:
        vocab_size: int = 267735
        cutoffs: List[int] = field(default_factory=lambda: [
                                   20000, 40000, 200000])
        d_model: int = 1024
        d_embed: int = 1024
        n_head: int = 16
        d_head: int = 64
        d_inner: int = 4096
        div_val: int = 4
        pre_lnorm: bool = False
        n_layer: int = 18
        mem_len: int = 1600
        clamp_len: int = 1000
        same_lenght: bool = True
        proj_share_all_but_first: bool = True
        attn_type: int = 0
        sample_softmax: int = -1
        adaptive: bool = True
        dropout: float = 0.1
        dropatt: float = 0
        untie_r: bool = True
        init: str = "normal"
        init_range: float = 0.01
        proj_init_std: float = 0.01
        init_std: float = 0.02
        layer_norm_epsilon: float = 1e-5

    @dataclass
    class XLM:

        vocab_size: int = 30145
        emb_dim: int = 2048
        n_layer: int = 12
        n_head: int = 16
        dropout: float = 0.1
        attention_dropout: float = 0.1
        gelu_activation: bool = True
        sinusoidal_embeddings: bool = False
        casual: bool = False
        asm: bool = False
        n_langs: int = 1
        use_lang_emb: bool = True
        max_position_embeddings: int = 512
        embed_init_std: float = 2048**-0.5
        init_std: int = 50257
        layer_norm_eps: float = 1e-12
        bos_index: int = 0
        eos_index: int = 1
        pad_index: int = 2
        unk_index: int = 3
        mask_index: int = 5
        is_encoder: bool = True
        summary_type: str = 'first'
        summary_use_proj: bool = True
        summary_proj_to_labels: bool = True
        summary_first_dropout: float = 0.1
        start_n_top: int = 5
        end_n_top: int = 5
        mask_token_id: int = 0
        lang_id: int = 1

    @dataclass
    class XLMProphetNet:

        activation_dropout: float = 0.1
        activation_function: str = 'gelu'
        vocab_size: int = 30522
        hidden_size: int = 1024
        num_encoder_layers: int = 12
        num_decoder_layers: int = 12
        num_encoder_attention_heads: int = 16
        num_decoder_attention_heads: int = 16
        decoder_ffn_dim: int = 4096
        encoder_ffn_dim: int = 4096
        dropout: float = 0.1
        attention_dropout: float = 0.1
        max_position_embeddings: int = 512
        init_std: float = 0.02
        add_cross_attention: bool = True
        is_encoder_decoder: bool = True
        ngram: int = 2
        num_buckets: int = 32
        relative_max_distance: int = 128
        disable_ngram_loss: bool = False
        eps: float = 0.0
        gradient_checkpointing: bool = False
        use_cache: bool = True

    @dataclass
    class XLMRoberta:

        vocab_size: int = 30522
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.1
        attention_probs_dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        type_vocab_size: int = 2
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        gradient_checkpointing: bool = False
        position_embedding_type: str = "absolute"
        use_cache = bool = True
        classifier_dropout: Optional[float] = None

    @dataclass
    class XLNet:
        vocab_size: int = 32000
        d_model: int = 1024
        n_layer: int = 24
        n_head: int = 16
        d_inner: int = 4096
        ff_activation: str = 'gelu'
        untie_r: bool = True
        attn_type: str = 'bi'
        initializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        dropout: float = 0.1
        bi_data: bool = False
        clamp_len: int = -1
        same_length: bool = False
        summary_type: str = "last"
        summary_use_proj: bool = True
        summary_proj_to_labels: bool = True
        summary_last_dropout: float = 0.1
        start_n_top: int = 5
        end_n_top: int = 5
        use_mems_eval: bool = True
        use_mems_train: bool = False


configs = {
    "Bert": BertConfig,
    'Albert': AlbertConfig,
    "Bart": BartConfig,
    "BigBird": BigBirdConfig,
    "BigBirdPegasus": BigBirdPegasusConfig,
    "Camembert": CamembertConfig,
    "Canine": CanineConfig,
    'CTRL': CTRLConfig,
    "ConvBert": ConvBertConfig,
    'Deberta': DebertaConfig,
    "DebertaV2": DebertaV2Config,
    "DistilBert": DistilBertConfig,
    "Electra": ElectraConfig,
    "Flaubert": FlaubertConfig,
    "Funnel": FunnelConfig,
    "GPT2": GPT2Config,
    "GPTNeo": GPTNeoConfig,
    "Hubert": HubertConfig,
    "IBert": IBertConfig,
    "LayoutLM": LayoutLMConfig,
    "LayoutLMv2": LayoutLMv2Config,
    "LED": LEDConfig,
    "Longformer": LongformerConfig,
    "Luke": LukeConfig,
    "Lxmert": LxmertConfig,
    "M2M100": M2M100Config,
    "Marian": MarianConfig,
    "MBart": MBartConfig,
    "MegatronBert": MegatronBertConfig,
    "MobileBert": MobileBertConfig,
    "MPNet": MPNetConfig,
    "MT5": MT5Config,
    "OpenAIGPT": OpenAIGPTConfig,
    "Pegasus": PegasusConfig,
    "ProphetNet": ProphetNetConfig,
    "Rag": RagConfig,
    "Reformer": ReformerConfig,
    "RemBert": RemBertConfig,
    "Roberta": RobertaConfig,
    "RoFormer": RoFormerConfig,
    "Splinter": SplinterConfig,
    "SqueezeBert": SqueezeBertConfig,
    "T5": T5Config,
    "Tapas": TapasConfig,
    "TransfoXL": TransfoXLConfig,
    "XLM": XLMConfig,
    "XLMProphetNet": XLMProphetNetConfig,
    "XLMRoberta": XLMRobertaConfig,
    "XLNet": XLNetConfig

}


def get_config(name, config):
    config_dict = vars(config)
    cons = configs[name]()
    for key, value in config_dict.items():
        setattr(cons, key, value)

    return cons
