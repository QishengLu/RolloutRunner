"""将 agent 结果写回本地 DB 的 EvaluationSample。"""
import datetime
import json
import logging

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from src.cost_metrics import build_cost_metrics
from src.dataset import EvaluationSample, get_engine
from src.runner import AgentResult

logger = logging.getLogger(__name__)


def write_result(
    *,
    result: AgentResult,
    exp_id: str,
    agent_type: str,
    model_name: str,
) -> bool:
    engine = get_engine()
    with Session(engine) as session:
        sample = session.get(EvaluationSample, result.sample_id)
        if sample is None:
            logger.error(f"Sample {result.sample_id} not found in DB")
            return False

        sample.response = result.output
        sample.trajectories = json.dumps(result.trajectory, ensure_ascii=False)
        sample.time_cost = result.time_cost
        sample.stage = "rollout"
        sample.exp_id = exp_id
        sample.agent_type = agent_type
        sample.model_name = model_name
        sample.updated_at = datetime.datetime.now()

        # 计算并存储 cost_metrics 到 meta
        cost_metrics = build_cost_metrics(
            trajectory=result.trajectory,
            usage=result.usage if result.usage else None,
            model=model_name,
            time_cost=result.time_cost,
        )
        meta = sample.meta or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        meta["cost_metrics"] = cost_metrics
        sample.meta = meta
        flag_modified(sample, "meta")

        session.add(sample)
        session.commit()

    logger.info(f"[sample {result.sample_id}] Written to DB (stage=rollout)")
    return True


def write_batch(
    results: list[AgentResult | None],
    exp_id: str,
    agent_type: str,
    model_name: str,
) -> tuple[int, int]:
    success, failure = 0, 0
    for result in results:
        if result is None:
            failure += 1
            continue
        ok = write_result(
            result=result, exp_id=exp_id, agent_type=agent_type, model_name=model_name
        )
        success += 1 if ok else 0
        failure += 0 if ok else 1
    return success, failure
