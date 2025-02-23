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
import asyncio
import logging
from typing import List, Tuple

from camel.types import ModelPlatformType

from oasis.inference.inference_thread import AsyncInferenceWorker

inference_log = logging.getLogger(name="inference")
inference_log.setLevel("DEBUG")

file_handler = logging.FileHandler("inference.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
inference_log.addHandler(file_handler)


class InferencerManager:
    def __init__(
        self,
        channel,
        model_type: str,
        model_path: str,
        stop_tokens: List[str],
        server_urls: List[str],
        max_workers: int = 20,
        max_concurrent_per_worker: int = 5
    ):
        self.channel = channel
        self.workers = []
        self.max_workers = max_workers
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        
        model_config = {
            "temperature": 0.0,
            "stop": stop_tokens,
            "max_tokens": 2048
        }
        
        for url in server_urls[:max_workers]:
            worker = AsyncInferenceWorker(
                server_url=url,
                model_type=model_type,
                model_platform_type=ModelPlatformType.VLLM,
                model_config=model_config,
                max_concurrent=max_concurrent_per_worker
            )
            self.workers.append(worker)

    async def run(self):
        consumers = [self.consume_requests(worker) for worker in self.workers]
        sender_task = asyncio.create_task(self.send_responses())
        
        try:
            await asyncio.gather(*consumers, sender_task)
        except asyncio.CancelledError:
            await self.stop()

    async def consume_requests(self, worker: AsyncInferenceWorker):
        while not self.stop_event.is_set():
            if not self.channel.receive_queue.empty():
                message_id, message = await self.channel.receive_from()
                task = asyncio.create_task(
                    worker.process_request(message_id, message))
                async with self.lock:
                    worker.pending_tasks[message_id] = task
            await asyncio.sleep(0.001)

    async def send_responses(self):
        while not self.stop_event.is_set():
            for worker in self.workers:
                done = set()
                for msg_id, task in worker.pending_tasks.items():
                    if task.done():
                        try:
                            result = await task
                            await self.channel.send_to(result)
                        except Exception as e:
                            inference_log.error(f"Send response failed: {str(e)}")
                        finally:
                            done.add(msg_id)
                
                async with self.lock:
                    for msg_id in done:
                        del worker.pending_tasks[msg_id]
            
            await asyncio.sleep(0.001)

    async def stop(self):
        self.stop_event.set()
        for worker in self.workers:
            while worker.pending_tasks:
                await asyncio.sleep(0.1)
