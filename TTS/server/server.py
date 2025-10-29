#!flask/bin/python
import argparse
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs

from flask import Flask, render_template, render_template_string, request, send_file

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")

    # --- NEW: option serveur pour la vitesse/rythme par défaut (VITS) ---
    # length_scale < 1.0 = plus rapide ; > 1.0 = plus lent
    parser.add_argument(
        "--length_scale_default",
        type=float,
        default=1.0,
        help="Default VITS length_scale. Smaller=faster, larger=slower.",
    )
    # --------------------------------------------------------------------

    # --- NEW: options serveur pour le contrôle de la variabilité à l'inférence ---
    # inference_noise_scale: variation prosodique globale au décodeur
    # inference_noise_scale_dp: variation sur le duration predictor (rythme local)
    # Par défaut None = ne pas forcer et laisser la valeur du modèle.
    parser.add_argument(
        "--inference_noise_scale_default",
        type=float,
        default=None,
        help="Default inference_noise_scale. Higher=more variation. None keeps model default.",
    )
    parser.add_argument(
        "--inference_noise_scale_dp_default",
        type=float,
        default=None,
        help="Default inference_noise_scale_dp (duration predictor). Higher=more timing variation. None keeps model default.",
    )
    # ------------------------------------------------------------------------------

    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)

use_multi_language = hasattr(synthesizer.tts_model, "num_languages") and (
    synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
)
language_manager = getattr(synthesizer.tts_model, "language_manager", None)

# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)

# --- NEW: helpers pour length_scale côté requête ---
# On lit un éventuel paramètre de requête/entête et on applique sur le modèle VITS.
# Acceptés: header "length-scale" ou "length_scale", champs GET/POST "length_scale".
def _read_length_scale_from_request() -> Union[None, float]:
    val = (
        request.headers.get("length-scale")
        or request.headers.get("length_scale")
        or request.values.get("length_scale")
    )
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None  # on ignore silencieusement si non numérique


def _apply_length_scale_temporarily(ls: Union[None, float]):
    """
    Applique length_scale sur tts_model si possible et renvoie un callback de reset.
    - Si ls est None: on applique la valeur par défaut serveur.
    - Si le modèle ne possède pas 'length_scale': on ne fait rien.
    """
    # valeur à appliquer pour cette synthèse
    target = args.length_scale_default if ls is None else ls

    if hasattr(synthesizer.tts_model, "length_scale"):
        # sauvegarde pour reset après synthèse
        old = synthesizer.tts_model.length_scale
        synthesizer.tts_model.length_scale = target

        def _reset():
            try:
                synthesizer.tts_model.length_scale = old
            except Exception:
                pass

        return _reset
    else:
        # pas de support length_scale sur ce modèle
        def _noop():
            return None

        return _noop
# ---------------------------------------------------

# --- NEW: helpers pour inference_noise_scale et inference_noise_scale_dp ---
# Lecture depuis headers/params et application temporaire avec reset après inférence.
def _read_float_from_request(*keys) -> Union[None, float]:
    """
    Lit la première clé disponible dans headers ou params et tente un float.
    Retourne None si absente ou invalide.
    """
    for k in keys:
        v = request.headers.get(k)
        if v is None or v == "":
            v = request.values.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                return None
    return None


def _apply_inference_noise_scale_temporarily(val: Union[None, float]):
    """
    Applique inference_noise_scale pour cette requête.
    - Si val est None: utilise --inference_noise_scale_default s'il est fourni, sinon ne force rien.
    """
    to_apply = val if val is not None else args.inference_noise_scale_default
    if to_apply is None:
        # rien à faire
        def _noop():
            return None
        return _noop

    if hasattr(synthesizer.tts_model, "inference_noise_scale"):
        old = synthesizer.tts_model.inference_noise_scale
        synthesizer.tts_model.inference_noise_scale = to_apply

        def _reset():
            try:
                synthesizer.tts_model.inference_noise_scale = old
            except Exception:
                pass
        return _reset
    else:
        def _noop():
            return None
        return _noop


def _apply_inference_noise_scale_dp_temporarily(val: Union[None, float]):
    """
    Applique inference_noise_scale_dp pour cette requête.
    - Si val est None: utilise --inference_noise_scale_dp_default s'il est fourni, sinon ne force rien.
    """
    to_apply = val if val is not None else args.inference_noise_scale_dp_default
    if to_apply is None:
        def _noop():
            return None
        return _noop

    if hasattr(synthesizer.tts_model, "inference_noise_scale_dp"):
        old = synthesizer.tts_model.inference_noise_scale_dp
        synthesizer.tts_model.inference_noise_scale_dp = to_apply

        def _reset():
            try:
                synthesizer.tts_model.inference_noise_scale_dp = old
            except Exception:
                pass
        return _reset
    else:
        def _noop():
            return None
        return _noop
# ---------------------------------------------------------------------------


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=use_multi_speaker,
        use_multi_language=use_multi_language,
        speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None,
        language_ids=language_manager.name_to_id if language_manager is not None else None,
        use_gst=use_gst,
    )


@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)

        # --- NEW: lecture et application temporaire du length_scale ---
        # Permet de contrôler la vitesse/rythme depuis la requête.
        req_ls = _read_length_scale_from_request()
        _reset_length_scale = _apply_length_scale_temporarily(req_ls)
        # --------------------------------------------------------------

        # --- NEW: lecture et application des bruits d'inférence ---
        # Headers/params acceptés:
        #  - inference-noise-scale, inference_noise_scale
        #  - inference-noise-scale-dp, inference_noise_scale_dp
        req_ins = _read_float_from_request("inference-noise-scale", "inference_noise_scale")
        req_ins_dp = _read_float_from_request("inference-noise-scale-dp", "inference_noise_scale_dp")
        _reset_ins = _apply_inference_noise_scale_temporarily(req_ins)
        _reset_ins_dp = _apply_inference_noise_scale_dp_temporarily(req_ins_dp)
        # -----------------------------------------------------------

        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        try:
            wavs = synthesizer.tts(text, speaker_name=speaker_idx, language_name=language_idx, style_wav=style_wav)
        finally:
            # --- NEW: on rétablit les valeurs précédentes après synthèse ---
            _reset_length_scale()
            _reset_ins()
            _reset_ins_dp()
            # ---------------------------------------------------------------
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
            # --- NEW: support length_scale en POST MaryTTS (optionnel) ---
            # Si un client envoie length_scale dans le form-url-encoded, on le lit ici.
            ls_str = data.get("length_scale", [None])[0]
            req_ls = float(ls_str) if ls_str not in (None, "") else None

            # --- NEW: support des bruits d'inférence en POST MaryTTS ---
            ins_str = data.get("inference_noise_scale", [None])[0]
            ins_dp_str = data.get("inference_noise_scale_dp", [None])[0]
            req_ins = float(ins_str) if ins_str not in (None, "") else None
            req_ins_dp = float(ins_dp_str) if ins_dp_str not in (None, "") else None
            # -----------------------------------------------------------
        else:
            text = request.args.get("INPUT_TEXT", "")
            # --- NEW: support length_scale en GET MaryTTS (optionnel) ---
            req_ls = _read_length_scale_from_request()
            # --- NEW: support des bruits d'inférence en GET MaryTTS ---
            req_ins = _read_float_from_request("inference-noise-scale", "inference_noise_scale")
            req_ins_dp = _read_float_from_request("inference-noise-scale-dp", "inference_noise_scale_dp")
            # ------------------------------------------------------------

        # --- NEW: application temporaire du length_scale ---
        _reset_length_scale = _apply_length_scale_temporarily(req_ls)
        # --- NEW: application temporaire des bruits d'inférence ---
        _reset_ins = _apply_inference_noise_scale_temporarily(req_ins)
        _reset_ins_dp = _apply_inference_noise_scale_dp_temporarily(req_ins_dp)
        # ---------------------------------------------------

        print(f" > Model input: {text}")
        try:
            wavs = synthesizer.tts(text)
        finally:
            # --- NEW: reset après synthèse ---
            _reset_length_scale()
            _reset_ins()
            _reset_ins_dp()
            # ---------------------------------
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()
