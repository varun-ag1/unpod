import asyncio
import os
import time
import traceback
from typing import Optional
from datetime import datetime, timedelta
from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run
from pydantic import BaseModel, Field
from super_services.libs.core.sms import send_sms_msg91
from super_services.libs.core.jsondecoder import sanitize_data_for_mongodb
from super_services.prefect_setup.deployments.utils import trigger_deployment
from super_services.prefect_setup.deployments.tag_concurrency import (
    create_run_concurrency_tag,
    delete_run_concurrency_tag,
)

HIGH_PRIORITY = "call-high-priority"


def generate_task_name():
    parameters = flow_run.parameters
    job = parameters["job"]
    if hasattr(job, "model_dump"):
        job = job.model_dump()
    task_id = job["task_id"]
    run_type = job.get("run_type", "normal")
    run_id = job.get("run_id")
    if run_id:
        return f"{run_type}-{run_id}-task-{task_id}"
    return f"{run_type}-task-{task_id}"


def generate_run_flow_name():
    parameters = flow_run.parameters
    job = parameters["job"]
    run_id = job.get("run_id")
    run_type = job.get("run_type")
    retry = job.get("retry", 0)
    return f"run-flow-{run_type}-{run_id}-{retry}"


def generate_sent_retry_sms_flow_name():
    parameters = flow_run.parameters
    task_id = parameters["task_id"]
    return f"sent_retry_sms-{task_id}"


def generate_reschedule_run_flow_name():
    parameters = flow_run.parameters
    job = parameters["job"]
    run_id = job.get("run_id")
    retry = job.get("retry", 0)
    return f"reschedule_run-{run_id}-retry-{retry}"


def generate_call_run_name():
    parameters = flow_run.parameters
    task_id = parameters["task_id"]
    run_id = parameters["run_id"]
    return f"call-task-{task_id}-{run_id}"


@task(
    description="Execute Call Activity",
    tags=["execute_call"],
    log_prints=True,
)
def execute_call_activity(
    task_id: str,
    agent_id: str,
    data: dict,
    run_id: str = None,
    instructions: str = None,
    run_type: str = None,
):
    """
    Full call execution task with Prefect tag-based concurrency.

    Uses RedisLockManager for dedup, locking, and scheduling —
    same pattern as the old _process_task_worker in voice_task_consumer.

    Lifecycle:
    1. Dedup check + distributed lock (via RedisLockManager)
    2. Atomic status transition (pending → processing)
    3. Max retries check
    4. Business hours check → reschedule via RedisLockManager
    5. Call dispatch via CallDispatcher
    6. Status update + dedup cache + webhook + thread update
    7. Call log + cost deduction
    8. Run status check
    9. Lock release (always, via finally)
    """
    from super_services.orchestration.task.task_service import TaskService
    from super_services.db.services.models.task import TaskModel
    from super_services.db.services.schemas.task import TaskStatusEnum
    from .call_dispatcher import CallDispatcher
    from .voice_task_consumer import RedisLockManager

    task_start_time = time.time()
    redis_manager = RedisLockManager(mode="prefect")

    # Step 1: Dedup check
    is_processed, reason = redis_manager._is_task_already_processed(task_id)
    if is_processed:
        print(f"Task {task_id} already processed (reason: {reason})")
        return {"status": "skipped", "data": {"reason": reason}}

    # Acquire distributed lock
    if not redis_manager._acquire_task_lock(task_id):
        print(f"Task {task_id} locked by another worker")
        return {"status": "skipped", "data": {"reason": "locked"}}

    task_service = TaskService()
    provider_name = CallDispatcher.get_provider_type(data)
    max_retries = int(os.getenv("MAX_CALL_RETRIES", "3"))
    task_obj: TaskModel.Meta.model = TaskModel.get(task_id=task_id)
    retry_attempt = getattr(task_obj, "retry_attempt", 0) if task_obj else 0
    counter_incremented = False

    try:
        # Step 2: Atomic status transition pending -> processing
        success, result = task_service.update_task_status_atomic(
            task_id,
            from_status=TaskStatusEnum.pending,
            to_status=TaskStatusEnum.processing,
        )
        if not success:
            print(f"Task {task_id} status already changed by another worker")
            return {"status": "skipped", "data": {"reason": "status_changed"}}

        # Step 3: Max retries check
        if retry_attempt >= max_retries:
            print(
                f"Task {task_id} exceeded max retries ({retry_attempt}), marking failed"
            )
            task_service.update_task_status(
                task_id,
                TaskStatusEnum.failed,
                {"error": f"Task exceeded max retries ({retry_attempt})"},
            )
            if task_obj:
                task_service.check_and_update_run_status(task_obj.run_id)
            return {"status": "failed", "data": {"error": "max_retries_exceeded"}}

        # Step 4: Business hours check
        phone_number = data.get("contact_number")
        if phone_number:
            enable_check = (
                os.getenv("ENABLE_BUSINESS_HOURS_CHECK", "true").lower() == "true"
            )
            if enable_check:
                from super_services.libs.core.timezone_utils import (
                    is_within_business_hours,
                )

                start_hour = int(os.getenv("BUSINESS_HOURS_START", "9"))
                end_hour = int(os.getenv("BUSINESS_HOURS_END", "20"))
                is_valid, next_available_time = is_within_business_hours(
                    phone_number, start_hour=start_hour, end_hour=end_hour
                )
                if not is_valid:
                    message = {
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "data": data,
                        "instructions": instructions,
                    }
                    scheduled_time_utc = redis_manager._schedule_task_for_later(
                        task_id, message, next_available_time
                    )
                    if scheduled_time_utc:
                        _reschedule_task_to_prefect(task_id, scheduled_time_utc)
                    print(
                        f"Task {task_id} outside business hours, scheduled for {next_available_time}"
                    )
                    return {
                        "status": "rescheduled",
                        "data": {"next": str(next_available_time)},
                    }

        # Increment provider concurrency counter
        new_counter = redis_manager._increment_redis_counter(provider_name)
        counter_incremented = True

        print(
            f"Processing task {task_id} for agent {agent_id}, "
            f"provider: {provider_name}, retry: {retry_attempt}, counter: {new_counter}"
        )

        # Step 5: Dispatch the call
        outgoing_calls_enabled = (
            os.getenv("OUTGOING_CALLS_ENABLED", "true").lower() == "true"
        )
        if not outgoing_calls_enabled:
            print(f"Outgoing calls disabled, simulating failure for {task_id}")
            response = {
                "status": "failed",
                "data": {"error": "Outgoing calls disabled via configuration."},
            }
        else:
            dispatcher = CallDispatcher()
            response = dispatcher.execute_call(
                agent_id=agent_id,
                task_id=task_id,
                data=data,
                instructions=instructions,
            )

        # Step 6: Update task status based on response
        if response and response.get("status") == "completed":
            print(f"Task {task_id} completed successfully")
            sanitized = sanitize_data_for_mongodb(response.get("data", {}))
            try:
                task_service.update_task_status(
                    task_id, TaskStatusEnum.completed, sanitized
                )
            except Exception as db_ex:
                print(
                    f"DB update failed for task {task_id}, sanitizing further: {db_ex}"
                )
                fallback_data = {
                    "error": "Data sanitization required",
                    "original_error": str(db_ex),
                    "data_summary": str(response.get("data", {}))[:1000],
                }
                task_service.update_task_status(
                    task_id, TaskStatusEnum.completed, fallback_data
                )

            # Dedup cache
            redis_manager._mark_task_in_dedup_cache(task_id)

            # Webhook + thread update
            try:
                from super_services.orchestration.webhook.webhook_handler import (
                    WebhookHandler,
                )

                from .call_dispatcher import _run_async

                webhook = WebhookHandler()
                _run_async(webhook.execute(task_id=task_id))
            except Exception as wh_ex:
                print(f"Webhook failed for task {task_id}: {wh_ex}")

            try:
                from super_services.voice.common.threads import update_thread

                update_thread(task_id, sanitized)
            except Exception as th_ex:
                print(f"Thread update failed for task {task_id}: {th_ex}")

            # Call log + cost deduction
            task_obj = TaskModel.get(task_id=task_id)
            resp_data = response.get("data", {})
            if task_obj and resp_data:
                try:
                    task_service.insert_call_log(response, resp_data, task_obj)
                    task_service.calculate_and_deduct_cost(resp_data, task_obj)
                except Exception as log_ex:
                    print(f"Failed to log/deduct cost for {task_id}: {log_ex}")

            if task_obj:
                task_service.check_and_update_run_status(task_obj.run_id)

        elif response and response.get("status") == "in_progress":
            sanitized = sanitize_data_for_mongodb(response.get("data", {}))
            task_service.update_task_status(
                task_id, TaskStatusEnum.in_progress, sanitized
            )

        else:
            print(f"Task {task_id} failed: {response}")
            sanitized = sanitize_data_for_mongodb(
                response.get("data", {}) if response else {"error": "No response"}
            )
            task_service.update_task_status(task_id, TaskStatusEnum.failed, sanitized)
            task_obj = TaskModel.get(task_id=task_id)
            if task_obj:
                task_service.check_and_update_run_status(task_obj.run_id)

        return response

    except Exception as ex:
        print(f"Exception processing task {task_id}: {ex}")
        traceback.print_exc()
        try:
            task_service.update_task_status(
                task_id,
                TaskStatusEnum.failed,
                sanitize_data_for_mongodb({"error": f"Exception: {str(ex)}"}),
            )
        except Exception:
            pass
        return {"status": "failed", "data": {"error": str(ex)}}

    finally:
        # Always release lock
        redis_manager._release_task_lock(task_id)
        # Decrement counter if we incremented it
        if counter_incremented:
            redis_manager._decrement_redis_counter(provider_name)
        task_duration_ms = (time.time() - task_start_time) * 1000
        print(f"Task {task_id} completed in {task_duration_ms:.0f}ms")


@flow(
    name="Execute Call Flow Consumer",
    description="Process Call jobs",
    flow_run_name=generate_call_run_name,
)
def execute_call_flow_consumer(
    run_id: str,
    task_id: str,
    agent_id: str,
    data: dict,
    instructions: str = None,
    model_config_data: dict = None,
    run_type=None,
    index=0,
):
    import random
    from .call_dispatcher import CallDispatcher

    provider_name = CallDispatcher.get_provider_type(data)
    run_tag = f"execute_{run_type or 'call'}"

    random_sleep = random.randint(3, 8) * (index + 1)
    if random_sleep > 20:
        random_sleep = random_sleep // 2
    time.sleep(random_sleep)

    execute_call_activity.with_options(
        name=f"execute_call_task_{task_id}",
        tags=[f"call_{provider_name}", run_tag],
    )(
        task_id=task_id,
        agent_id=agent_id,
        data=data,
        run_id=run_id,
        instructions=instructions,
        run_type=run_type,
    )


def _reschedule_task_to_prefect(task_id, scheduled_time_utc):
    """Schedule a task for later execution via Prefect deployment."""
    from prefect.states import Scheduled
    from .call_dispatcher import _run_async

    _run_async(
        trigger_deployment(
            "Execute-Task",
            {"job": {"task_id": task_id, "retry": 0, "run_type": "call"}},
            state=Scheduled(scheduled_time=scheduled_time_utc),
        )
    )


@flow(
    name="Check Reschedule Run",
    description="Check and reschedule run flow",
    flow_run_name=generate_reschedule_run_flow_name,
)
async def check_reschedule_run(job):
    run_id = job.get("run_id")
    retry = job.get("retry", 0)
    run_type = job.get("run_type")
    logger = get_run_logger()
    from super_services.orchestration.task.task_service import TaskService

    task_service = TaskService()
    tasks = task_service.get_pending_tasks(run_id) or []
    logger.info(f"Reschedule run {run_id} with {len(tasks)} tasks, retry {retry}")
    if tasks and retry < 3:
        from prefect.states import Scheduled

        next_5min = datetime.utcnow() + timedelta(minutes=5)
        flow_name = f"run-flow-{run_type}-{run_id}-{retry}"
        await trigger_deployment(
            "Execute Run",
            {"job": {"run_id": run_id, "retry": retry + 1, "run_type": run_type}},
            state=Scheduled(scheduled_time=next_5min),
            name=flow_name,
        )
    return True


@flow(
    name="Process Run Flow",
    description="Process Run jobs",
    flow_run_name=generate_run_flow_name,
)
async def process_run_flow(job):
    from super_services.orchestration.task.task_service import TaskService

    run_id = job.get("run_id")
    retry = job.get("retry", 0)
    logger = get_run_logger()

    logger.info(f"Processing run {run_id} with retry {retry}")

    task_service = TaskService()
    run_status = task_service.get_run_status(run_id)
    run_obj = run_status.get("run") if run_status else None
    if not run_obj:
        logger.error(f"Run {run_id} not found")
        return
    if run_obj.get("status") == "completed":
        logger.info(f"Run {run_id} is already completed")
        return
    run_type = run_obj.get("run_type")
    if run_type not in ["call"]:
        logger.info(f"Run {run_id} is of type {run_type}, skipping processing")
        return
    logger.info(f"Run {run_id} is in {run_obj.get('status')}, processing tasks")
    tasks = task_service.get_pending_tasks(run_id, run_status=run_obj.get("status"))
    if not tasks:
        logger.info(f"No pending tasks for run {run_id}")
        return

    logger.info(f"Processing run {run_id} with {len(tasks)} tasks")

    # Create dynamic concurrency tag for this run
    run_tag = f"run_{run_id}"
    RUN_WISE_CONCURRENCY = os.environ.get("RUN_WISE_CONCURRENCY", 10)
    run_wise_concurrency = min(int(RUN_WISE_CONCURRENCY), len(tasks))
    await create_run_concurrency_tag(run_id, concurrency_limit=run_wise_concurrency)
    logger.info(
        f"Created concurrency tag '{run_tag}' with limit {run_wise_concurrency}"
    )

    triggered = 0
    try:
        for index, task in enumerate(tasks):
            task_id = task.get("task_id")
            task_flow_name = f"{run_type}-{run_id}-task-{task_id}"
            extra_kargs = {}
            if len(tasks) < 5:
                extra_kargs["work_queue_name"] = HIGH_PRIORITY
            try:
                await trigger_deployment(
                    "Execute-Task",
                    {
                        "job": {
                            "task_id": task_id,
                            "run_type": run_type,
                            "run_id": run_id,
                        }
                    },
                    tags=[run_tag],
                    name=task_flow_name,
                    **extra_kargs,
                )
                triggered += 1
                logger.info(
                    f"Triggered task {task_id} for run {run_id} index, {index+1}"
                )
            except Exception as ex:
                logger.error(f"Failed to trigger task {task_id} for run {run_id}: {ex}")
            await asyncio.sleep(0.5)
    finally:
        logger.info(f"Triggered {triggered}/{len(tasks)} tasks for run {run_id}")


class TaskJob(BaseModel):
    task_id: str
    run_type: str
    retry: Optional[int] = Field(default=0)
    run_id: Optional[str] = Field(default=None)


@flow(
    name="Process Task Flow",
    description="Process Task job",
    flow_run_name=generate_task_name,
    log_prints=True,
)
def process_task_flow(job: TaskJob):
    """
    Thin flow wrapper — fetches task from DB, then delegates to
    execute_call_activity @task with Prefect tag-based concurrency.

    All execution logic (lock, dedup, status, dispatch, webhook, etc.)
    lives in execute_call_activity.
    """
    from super_services.orchestration.task.task_service import TaskService
    from .call_dispatcher import CallDispatcher

    logger = get_run_logger()

    task_service = TaskService()
    task_data = task_service.get_task(job.task_id)
    if not task_data:
        logger.error(f"Task {job.task_id} not found")
        return

    run_id = task_data.get("run_id")
    task_id = task_data.get("task_id")
    agent_id = task_data.get("assignee")
    data = task_data.get("input") or {}
    instructions = task_data.get("task", {}).get("objective")
    provider_name = CallDispatcher.get_provider_type(data)

    logger.info(
        f"Processing task {task_id} for run {run_id}, provider: {provider_name}"
    )

    run_tag = f"execute_{job.run_type or 'call'}"

    execute_call_activity.with_options(
        name=f"execute_call_task_{task_id}",
        tags=[f"call_{provider_name}", run_tag],
    )(
        task_id=task_id,
        agent_id=agent_id,
        data=data,
        run_id=run_id,
        instructions=instructions,
        run_type=job.run_type,
    )


@flow(
    name="Run Cleanup Flow",
    description="Run Cleanup Flow",
    flow_run_name=generate_run_flow_name,
)
async def process_run_cleanup_flow(job):
    from super_services.orchestration.task.task_service import TaskService

    run_id = job.get("run_id")
    logger = get_run_logger()

    task_service = TaskService()
    run_obj = task_service.get_run(run_id)
    if not run_obj:
        logger.info(f"Run {run_id} not found for cleanup")
        return
    run_status = run_obj.get("status")
    if run_status in ["completed", "failed", "partially_completed"]:
        logger.info(f"Starting cleanup for run {run_id} with status {run_status}")
        try:
            await delete_run_concurrency_tag(run_id)
        except Exception as ex:
            logger.error(f"Error deleting concurrency tag for run {run_id}: {str(ex)}")
    else:
        logger.info(f"Run {run_id} is in {run_status}, skipping cleanup")


@flow(
    name="Send Retry SMS",
    description="Send Retry SMS to user",
    flow_run_name=generate_sent_retry_sms_flow_name,
)
async def sent_retry_sms(task_id: str, data: dict):
    from super_services.orchestration.task.task_service import TaskService
    from super_services.db.services.models.task import CallRetrySmsModel

    logger = get_run_logger()
    task_service = TaskService()
    task = task_service.get_task(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return
    temp_id = data.get("temp_id")
    contact_number = data.get("contact_number")
    kargs = data.get("kargs", {})
    try:
        res = await send_sms_msg91(contact_number, temp_id, **kargs)
        logger.info(f"Res - {res.text}")
        logger.info(f"Sent retry sms to {contact_number} for task {task_id}")
        CallRetrySmsModel._get_collection().update_one(
            {"task_id": task_id, "status": "processing"},
            {"$set": {"status": "completed", "response": res.text}},
        )
    except Exception as ex:
        logger.error(f"Failed to send SMS for task {task_id}: {ex}")
        CallRetrySmsModel._get_collection().update_one(
            {"task_id": task_id, "status": "processing"},
            {"$set": {"status": "failed", "response": str(ex)}},
        )
    return True
