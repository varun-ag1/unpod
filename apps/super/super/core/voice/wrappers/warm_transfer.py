from livekit.agents.beta.workflows import WarmTransferTask
import asyncio
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.job import get_job_context
from livekit.agents.log import logger
from livekit.agents import utils

class WarmTransferAgentTask(WarmTransferTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def on_enter(self) -> None:
        job_ctx = get_job_context()
        self._caller_room = job_ctx.room

        # start the background audio
        if self._hold_audio is not None:
            await self._background_audio.start(room=self._caller_room)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        self._set_io_enabled(False)

        try:
            dial_human_agent_task = asyncio.create_task(self._dial_human_agent())
            done, _ = await asyncio.wait(
                (dial_human_agent_task, self._human_agent_failed_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if dial_human_agent_task not in done:
                raise RuntimeError()

            self._human_agent_sess = dial_human_agent_task.result()
            try:
                await self._human_agent_sess.say("hello i have a caller on line")
            except Exception as e:
                logger.error(f"unable to use session.say {str(e)}")

        except Exception:
            logger.exception("could not dial human agent")
            self._set_result(ToolError("could not dial human agent"))
            return

        finally:
            await utils.aio.cancel_and_wait(dial_human_agent_task)
