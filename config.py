import os

import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

INITIAL_PEERS = os.getenv("INITIAL_PEERS", None)
if INITIAL_PEERS is not None:
    INITIAL_PEERS = INITIAL_PEERS.split(":")
else:
    INITIAL_PEERS = PUBLIC_INITIAL_PEERS
print("INITIAL_PEERS",INITIAL_PEERS)
default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    "Llama 2": [

        ModelConfig(
            ModelBackendConfig(repository
="VAGOsolutions/SauerkrautLM-Mixtral-8x7B
-Instruct"),
            ModelFrontendConfig(
                name="SauerkrautLM-Mixtra
l-8x7B-Instruct",
                model_card="https://huggi
ngface.co/VAGOsolutions/SauerkrautLM-Mixt
ral-8x7B-Instruct",
                license="https://chooseal
icense.com/licenses/apache-2.0",
            ),
            default_chat_config,
        ),
        
        # ModelConfig(
        #     ModelBackendConfig(repository="petals-team/StableBeluga2", aliases=["stabilityai/StableBeluga2"]),
        #     ModelFrontendConfig(
        #         name="Stable Beluga 2 (70B)",
        #         model_card="https://huggingface.co/stabilityai/StableBeluga2",
        #         license="https://huggingface.co/stabilityai/StableBeluga2/blob/main/LICENSE.txt",
        #     ),
        #     default_chat_config,
        # ),
        # ModelConfig(
        #     ModelBackendConfig(repository="Maykeye/TinyLLama-v0"),
        #     ModelFrontendConfig(
        #         name="TinyLlama-v0",
        #         model_card="https://huggingface.co/Maykeye/TinyLLama-v0",
        #         license="https://choosealicense.com/licenses/apache-2.0",
        #     ),
        #     default_chat_config,
        # ),
        # ModelConfig(
        #      ModelBackendConfig(repository="SanjiWatsuki/TinyMixtral-32x248M"),
        #      ModelFrontendConfig(
        #          name="TinyMixtral-32x248M",
        #          model_card="https://huggingface.co/SanjiWatsuki/TinyMixtral-32x248M",
        #          license="https://choosealicense.com/licenses/apache-2.0",
        #      ),
        #      default_chat_config,
        # ),

        # ModelConfig(
        #     ModelBackendConfig(repository="petals-team/StableBeluga2", aliases=["stabilityai/StableBeluga2"]),
        #     ModelFrontendConfig(
        #         name="Stable Beluga 2 (70B)",
        #         model_card="https://huggingface.co/stabilityai/StableBeluga2",
        #         license="https://huggingface.co/stabilityai/StableBeluga2/blob/main/LICENSE.txt",
        #     ),
        #     default_chat_config,
        # ),
        ModelConfig(
            ModelBackendConfig(repository="Maykeye/TinyLLama-v0"),
            ModelFrontendConfig(
                name="TinyLlama-v0",
                model_card="https://huggingface.co/Maykeye/TinyLLama-v0",
                license="https://choosealicense.com/licenses/apache-2.0",
            ),
            default_chat_config,
        ),
        ModelConfig(
            ModelBackendConfig(repository="SanjiWatsuki/TinyMixtral-32x248M"),
            ModelFrontendConfig(
                name="TinyMixtral-32x248M",
                model_card="https://huggingface.co/SanjiWatsuki/TinyMixtral-32x248M",
                license="https://choosealicense.com/licenses/apache-2.0",
            ),
            default_chat_config,
        ),

    ],
    # "Falcon": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="tiiuae/falcon-180B-chat", public_api=False),
    #         ModelFrontendConfig(
    #             name="Falcon 180B-Chat",
    #             model_card="https://huggingface.co/tiiuae/falcon-180B-chat",
    #             license="https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt",
    #         ),
    #         ModelChatConfig(
    #             max_session_length=8192,
    #             sep_token="\n",
    #             stop_token="\n",
    #             extra_stop_sequences=["<|endoftext|>", "\nFalcon:", " Falcon:", "\nUser:", " User:", "###"],
    #             generation_params=dict(do_sample=1, temperature=0.75, top_p=0.9, repetition_penalty=1.2),
    #         ),
    #     ),
    # ],
    # "Llama": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="huggyllama/llama-65b", adapter="timdettmers/guanaco-65b"),
    #         ModelFrontendConfig(
    #             name="Guanaco-65B",
    #             model_card="https://huggingface.co/timdettmers/guanaco-65b",
    #             license="https://huggingface.co/timdettmers/guanaco-65b",
    #         ),
    #         default_chat_config,
    #     ),
    #     ModelConfig(
    #         ModelBackendConfig(repository="huggyllama/llama-65b"),
    #         ModelFrontendConfig(
    #             name="Llama-65B",
    #             model_card="https://github.com/facebookresearch/llama/blob/llama_v1/MODEL_CARD.md",
    #             license="https://bit.ly/llama-license",
    #         ),
    #         default_chat_config,
    #     ),
    # ],
    # "BLOOM": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="bigscience/bloomz"),
    #         ModelFrontendConfig(
    #             name="BLOOMZ-176B",
    #             model_card="https://huggingface.co/bigscience/bloomz",
    #             license="https://bit.ly/bloom-license",
    #         ),
    #         ModelChatConfig(
    #             max_session_length=2048,
    #             sep_token="\n\n",
    #             stop_token="</s>",
    #             extra_stop_sequences=["\n\nHuman"],
    #             generation_params=default_chat_config.generation_params,
    #         ),
    #     ),
    # ],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
