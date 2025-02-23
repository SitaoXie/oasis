# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import logging
import asyncio
from typing import List, Tuple

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType

thread_log = logging.getLogger(name="inference.thread")
thread_log.setLevel("DEBUG")


class AsyncInferenceWorker:
    def __init__(
        self,
        server_url: str,
        model_type: str,
        model_platform_type: ModelPlatformType,
        model_config: dict,
        max_concurrent: int = 20
    ):
        self.server_url = server_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_platform=model_platform_type,
            model_type=model_type,
            model_config_dict=model_config,
            url="vllm",
            api_key=server_url,
        )
        self.pending_tasks = {}

    async def process_request(self, message_id: int, message: str) -> Tuple[int, str]:
        async with self.semaphore:
            try:
                response = await self.model_backend.async_run(message)
                return (message_id, response.choices[0].message.content)
            except Exception as e:
                thread_log.error(f"Request failed: {str(e)}")
                return (message_id, "No response")
