"""从本地 DB 文件读取待测评样本（DB 由 RCAgentEval 预处理后复制过来）。"""
import datetime
import os
from typing import Any

from sqlalchemy import Column
from sqlmodel import JSON, Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel


class UTUBaseModel(BaseModel):
    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UTUBaseModel":
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class EvaluationSample(UTUBaseModel, SQLModel, table=True):
    __tablename__ = "evaluation_data"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime.datetime | None = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime | None = Field(default_factory=datetime.datetime.now)

    # 1) base info
    dataset: str = ""
    dataset_index: int | None = Field(default=None)
    source: str = ""
    raw_question: str = ""
    level: int | None = 0
    augmented_question: str | None = ""
    correct_answer: str | None = ""
    file_name: str | None = ""
    meta: Any | None = Field(default=None, sa_column=Column(JSON))
    # 2) rollout
    trace_id: str | None = Field(default=None)
    trace_url: str | None = Field(default=None)
    response: str | None = Field(default=None)
    time_cost: float | None = Field(default=None)
    trajectory: str | None = Field(default=None)
    trajectories: str | None = Field(default=None)
    # 3) judgement
    extracted_final_answer: str | None = Field(default=None)
    judged_response: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    correct: bool | None = Field(default=None)
    confidence: float | None = Field(default=None)
    # id
    exp_id: str = Field(default="default")
    agent_type: str | None = Field(default=None, index=True)
    model_name: str | None = Field(default=None, index=True)
    stage: str = "init"

    def model_dump(self, *args, **kwargs):
        keys = [
            "exp_id",
            "agent_type",
            "model_name",
            "dataset",
            "dataset_index",
            "source",
            "level",
            "raw_question",
            "correct_answer",
            "file_name",
            "stage",
            "trace_id",
            "response",
            "time_cost",
            "trajectory",
            "trajectories",
            "judged_response",
            "correct",
            "confidence",
        ]
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


def get_engine():
    db_url = os.environ.get("UTU_DB_URL")
    if not db_url:
        raise ValueError("UTU_DB_URL 环境变量未设置（示例：sqlite:///./eval.db）")
    return create_engine(db_url)


def load_samples(source_exp_id: str) -> list[EvaluationSample]:
    """加载指定 exp_id 下 stage='init' 的样本。"""
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(EvaluationSample).where(
            EvaluationSample.exp_id == source_exp_id,
            EvaluationSample.stage == "init",
        )
        return list(session.exec(stmt).all())
