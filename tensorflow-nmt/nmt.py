# -*- coding: utf-8 -*-

#####################################################################
# File Name:  nmt.py
# Author: shenming
# Created Time: Wed Jan  1 23:11:54 2020
#####################################################################

"""
This is the main program entry point
"""

import os
import sys
import argparse

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorboard.plugins.hparams import api as hp

import train
from utils import misc_utils as utils
from utils import hparams

utils.check_tensorflow_version()

def add_arguments(parser):
    """Build ArgumentParser."""
    # 注册一个 bool 类型, 处理方式就是后面的函数
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units", type=int, default=32,
                        help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--encoder_type", type=str, default="uni", help="""\
    uni | bi | gnmt.
    For bi, we build num_encoder_layers/2 bi-directional layers.
    For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
    uni-directional layers.\
    """)
    # 此处用到了 bool 类型
    parser.add_argument("--residual", type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to add residual connections.")
    parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                        default=True,
                        help="Whether to use time-major mode for dynamic RNN.")
    parser.add_argument("--num_embeddings_partitions", type=int,
                        default=0,
                        help="Number of partitions for embedding vars.")

    # attention mechanisms
    parser.add_argument("--attention", type=str, default="", help="""\
    luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
    attention\
      """)
    parser.add_argument(
        "--attention_architecture",
        type=str,
        default="standard",
        help="""\
        standard | gnmt | gnmt_v2.
        standard: use top layer to compute attention.
        gnmt: GNMT style of computing attention, use previous bottom layer to
            compute attention.
        gnmt_v2: similar to gnmt, but use current bottom layer to compute
        attention.\
      """)
    parser.add_argument(
        "--output_attention", type="bool", nargs="?", const=True,
        default=True,
        help="""\
        Only used in standard attention_architecture. Whether use attention as
        the cell output at each timestep.
        .\
        """)
    parser.add_argument(
        "--pass_hidden_state", type="bool", nargs="?", const=True,
        default=True,
        help="""\
        Whether to pass encoder's hidden state to decoder when using an attention
        based model.\
      """)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
    How to warmup learning rates. Options include:
    t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
    exponentiate until the specified lr.\
    """)
    parser.add_argument(
        "--decay_scheme", type=str, default="", help="""\
        How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
        for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
        for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
        for 10 times before finishing.\
        """)

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))

    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))

    # data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix, e.g., en.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="Target suffix, e.g., de.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
    Vocab prefix, expect files with src/tgt suffixes.\
    """)
    parser.add_argument("--embed_prefix", type=str, default=None, help="""\
    Pretrained embedding prefix, expect files with src/tgt suffixes.
    The embedding files should be Glove formated txt files.\
    """)
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                        default=False,
                        help="""\
                        Whether to use the source vocab and embeddings for both source and
                        target.\
      """)
    parser.add_argument("--check_special_token", type="bool", default=True,
                        help="""\
                        Whether check special sos, eos, unk tokens exist in the
                        vocab files.\
                        """)

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""\
                        Max length of tgt sequences during inference.  Also use to restrict the
                        maximum decoding length.\
                        """)

    # Default settings works well (rarely need to change)
    parser.add_argument("--unit_type", type=str, default="lstm",
                        help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget_bias", type=float, default=1.0,
    help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--num_sampled_softmax", type=int, default=0,
                        help=("Use sampled_softmax_loss if > 0."
                              "Otherwise, use full softmax loss."))

    # SPM
    parser.add_argument("--subword_option", type=str, default="",
                        choices=["", "bpe", "spm"],
                        help="""\
                        Set to bpe or spm to activate subword desegmentation.\
                        """)

    # Experimental encoding feature.
    parser.add_argument("--use_char_encode", type="bool", default=False,
                        help="""\
                        Whether to split each word or bpe into character, and then
                        generate the word-level representation from the character
                        reprentation.
                        """)

    # Misc
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of gpus in each worker.")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--metrics", type=str, default="bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,rouge,accuracy)"))
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
                        How many training steps to do per external evaluation.  Automatically set
                        based on data if None.\
                        """)
    parser.add_argument("--scope", type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from FLAGS."))
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                        const=True, default=False,
                        help="Override loaded hparams with values specified")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                        const=True, default=False, help=("""\
                        Average the last N checkpoints for external evaluation.
                        N can be controlled by setting --num_keep_ckpts.\
                        """))
    parser.add_argument("--language_model", type="bool", nargs="?",
                        const=True, default=False,
                        help="True to train a language model, ignoring encoder")

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--inference_list", type=str, default=None,
                        help=("A comma-separated list of sentence indices "
                              "(0-based) to decode."))
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
                        Reference file to compute evaluation scores (if provided).\
      """))

    # Advanced inference arguments
    parser.add_argument("--infer_mode", type=str, default="greedy",
                        choices=["greedy", "sample", "beam_search"],
                        help="Which type of decoder to use during inference.")
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
                        beam width when using beam search decoder. If 0 (default), use standard
                        decoder with greedy helper.\
                        """))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--coverage_penalty_weight", type=float, default=0.0,
                        help="Coverage penalty for beam search.")
    parser.add_argument("--sampling_temperature", type=float,
                        default=0.0,
                        help=("""\
                        Softmax sampling temperature for inference decoding, 0.0 means greedy
                        decoding. This option is ignored when using beam search.\
                        """))
    parser.add_argument("--num_translations_per_input", type=int, default=1,
                        help=("""\
                        Number of translations generated for each sentence. This is only used for
                        inference.\
                        """))

    # Job info
    parser.add_argument("--jobid", type=int, default=0,
                        help="Task id of the worker.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers (inference only).")
    parser.add_argument("--num_inter_threads", type=int, default=0,
                        help="number of inter_op_parallelism_threads")
    parser.add_argument("--num_intra_threads", type=int, default=0,
                        help="number of intra_op_parallelism_threads")

def create_hparams(flags):
    """Create training hparams."""
    return hparams.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        out_dir=flags.out_dir,

      # Networks
        num_units=flags.num_units,
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_partitions=flags.num_embeddings_partitions,

      # Attention mechanisms
        attention=flags.attention,
        attention_architecture=flags.attention_architecture,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,

      # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        num_sampled_softmax=flags.num_sampled_softmax,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,

        # Advanced inference arguments
        infer_mode=flags.infer_mode,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        coverage_penalty_weight=flags.coverage_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,
        use_char_encode=flags.use_char_encode,

      # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(","),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        language_model=flags.language_model,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
    )


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    # TODO
    # train_fn(default_hparams)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    main(unparsed)
    # tfv1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
